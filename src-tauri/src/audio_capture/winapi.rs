use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc, Mutex,
};
use std::thread;
use webrtc::media::Sample;
use webrtc::track::track_local::track_local_static_sample::TrackLocalStaticSample;
use windows::core::*;
use windows::Win32::Foundation::*;
use windows::Win32::Media::Audio::*;
use windows::Win32::System::Com::*;

const SAMPLE_RATE: u32 = 48000;
const BUFFER_SIZE: usize = 1920; // 40ms

#[derive(Clone)]
pub struct AudioData {
    pub mic: Vec<i16>,
    pub system: Vec<i16>,
    pub timestamp: u64,
}

pub struct AudioMixer {
    mic_volume: f32,
    system_volume: f32,
}

impl AudioMixer {
    pub fn new() -> Self {
        Self {
            mic_volume: 1.0,
            system_volume: 1.0,
        }
    }

    pub fn mix(&self, mic: &[i16], system: &[i16]) -> Vec<i16> {
        let len = mic.len().max(system.len());
        let mut result = Vec::with_capacity(len);

        for i in 0..len {
            let m = if i < mic.len() {
                mic[i] as f32 * self.mic_volume
            } else {
                0.0
            };
            let s = if i < system.len() {
                system[i] as f32 * self.system_volume
            } else {
                0.0
            };
            result.push((m + s).clamp(-32768.0, 32767.0) as i16);
        }
        result
    }

    pub fn set_mic_volume(&mut self, vol: f32) {
        self.mic_volume = vol.clamp(0.0, 2.0);
    }
    pub fn set_system_volume(&mut self, vol: f32) {
        self.system_volume = vol.clamp(0.0, 2.0);
    }
}

// 音频捕获线程
fn audio_capture_thread(
    sender: mpsc::Sender<AudioData>,
    running: Arc<AtomicBool>,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    unsafe {
        CoInitializeEx(None, COINIT_MULTITHREADED).ok();

        let enumerator: IMMDeviceEnumerator =
            CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)?;

        // 麦克风
        let mic_device = enumerator.GetDefaultAudioEndpoint(eCapture, eConsole)?;
        let mic_client: IAudioClient = mic_device.Activate(CLSCTX_ALL, None)?;
        let mic_format = mic_client.GetMixFormat()?;
        mic_client.Initialize(AUDCLNT_SHAREMODE_SHARED, 0, 10000000, 0, mic_format, None)?;
        let mic_capture: IAudioCaptureClient = mic_client.GetService()?;

        // 系统音频
        let sys_device = enumerator.GetDefaultAudioEndpoint(eRender, eConsole)?;
        let sys_client: IAudioClient = sys_device.Activate(CLSCTX_ALL, None)?;
        let sys_format = sys_client.GetMixFormat()?;
        sys_client.Initialize(
            AUDCLNT_SHAREMODE_SHARED,
            AUDCLNT_STREAMFLAGS_LOOPBACK,
            10000000,
            0,
            sys_format,
            None,
        )?;
        let sys_capture: IAudioCaptureClient = sys_client.GetService()?;

        let mut timestamp = 0u64;
        let mut capture_started = false;

        while running.load(Ordering::Relaxed) {
            thread::sleep(std::time::Duration::from_millis(20));

            // 延迟启动音频流直到需要时
            if !capture_started {
                mic_client.Start()?;
                sys_client.Start()?;
                capture_started = true;
            }

            let mut mic_data = Vec::new();
            let mut sys_data = Vec::new();

            // 捕获麦克风
            if let Ok(packet_len) = mic_capture.GetNextPacketSize() {
                if packet_len > 0 {
                    let mut buffer = std::ptr::null_mut();
                    let mut frames = 0u32;
                    let mut flags = 0u32;

                    if mic_capture
                        .GetBuffer(&mut buffer, &mut frames, &mut flags, None, None)
                        .is_ok()
                    {
                        let data =
                            std::slice::from_raw_parts(buffer as *const i16, frames as usize * 2);
                        mic_data.extend_from_slice(data);
                        mic_capture.ReleaseBuffer(frames).ok();
                    }
                }
            }

            // 捕获系统音频
            if let Ok(packet_len) = sys_capture.GetNextPacketSize() {
                if packet_len > 0 {
                    let mut buffer = std::ptr::null_mut();
                    let mut frames = 0u32;
                    let mut flags = 0u32;

                    if sys_capture
                        .GetBuffer(&mut buffer, &mut frames, &mut flags, None, None)
                        .is_ok()
                    {
                        let data =
                            std::slice::from_raw_parts(buffer as *const i16, frames as usize * 2);
                        sys_data.extend_from_slice(data);
                        sys_capture.ReleaseBuffer(frames).ok();
                    }
                }
            }

            // 发送数据
            if !mic_data.is_empty() || !sys_data.is_empty() {
                let audio = AudioData {
                    mic: mic_data,
                    system: sys_data,
                    timestamp,
                };
                if sender.send(audio).is_err() {
                    break;
                }
                timestamp += 20;
            }
        }

        // 停止音频流
        if capture_started {
            mic_client.Stop().ok();
            sys_client.Stop().ok();
        }
    }
    Ok(())
}

struct TrackInfo {
    track: Arc<TrackLocalStaticSample>,
    active: AtomicBool,
}

pub struct WebRTCAudioSystem {
    mixer: Arc<Mutex<AudioMixer>>,
    tracks: Arc<Mutex<HashMap<String, TrackInfo>>>,
    peer_volumes: Arc<Mutex<HashMap<String, f32>>>,
    capture_running: Arc<AtomicBool>,
    processing_running: Arc<AtomicBool>,
    _capture_handle: Option<thread::JoinHandle<()>>,
    _processing_handle: Option<tokio::task::JoinHandle<()>>,
}

impl WebRTCAudioSystem {
    pub fn new() -> Self {
        let mixer = Arc::new(Mutex::new(AudioMixer::new()));
        let tracks = Arc::new(Mutex::new(HashMap::new()));
        let peer_volumes = Arc::new(Mutex::new(HashMap::new()));
        let capture_running = Arc::new(AtomicBool::new(false));
        let processing_running = Arc::new(AtomicBool::new(false));

        Self {
            mixer,
            tracks,
            peer_volumes,
            capture_running,
            processing_running,
            _capture_handle: None,
            _processing_handle: None,
        }
    }

    /// 启动音频捕获和处理
    pub fn start_capture(&mut self) -> std::result::Result<(), Box<dyn std::error::Error>> {
        if self.capture_running.load(Ordering::Relaxed) {
            return Ok(()); // 已经启动
        }

        let (tx, rx) = mpsc::channel::<AudioData>();

        // 启动音频捕获线程
        let capture_running = Arc::clone(&self.capture_running);
        capture_running.store(true, Ordering::Relaxed);

        let capture_running_clone = Arc::clone(&capture_running);
        let capture_handle = thread::spawn(move || {
            if let Err(e) = audio_capture_thread(tx, capture_running_clone) {
                eprintln!("Audio capture error: {}", e);
            }
        });

        // 启动音频处理任务
        let processing_running = Arc::clone(&self.processing_running);
        processing_running.store(true, Ordering::Relaxed);

        let mixer_clone = Arc::clone(&self.mixer);
        let tracks_clone = Arc::clone(&self.tracks);
        let processing_running_clone = Arc::clone(&processing_running);

        let processing_handle = tokio::spawn(async move {
            while processing_running_clone.load(Ordering::Relaxed) {
                if let Ok(audio) = rx.recv() {
                    // 混音
                    let mixed = if let Ok(m) = mixer_clone.lock() {
                        m.mix(&audio.mic, &audio.system)
                    } else {
                        continue;
                    };

                    // 收集活跃的tracks，避免跨await持有锁
                    let active_tracks = if let Ok(tracks_guard) = tracks_clone.lock() {
                        tracks_guard
                            .values()
                            .filter(|info| info.active.load(Ordering::Relaxed))
                            .map(|info| Arc::clone(&info.track))
                            .collect::<Vec<_>>()
                    } else {
                        continue;
                    };

                    // 释放锁后再发送数据
                    if !active_tracks.is_empty() {
                        let sample_data = mixed.iter().flat_map(|&x| x.to_le_bytes()).collect();
                        let sample = Sample {
                            data: sample_data,
                            duration: std::time::Duration::from_millis(20),
                            ..Default::default()
                        };

                        for track in active_tracks {
                            let _ = track.write_sample(&sample).await;
                        }
                    }
                } else {
                    break; // 通道关闭
                }
            }
        });

        self._capture_handle = Some(capture_handle);
        self._processing_handle = Some(processing_handle);

        Ok(())
    }

    /// 停止音频捕获
    pub fn stop_capture(&mut self) {
        self.capture_running.store(false, Ordering::Relaxed);
        self.processing_running.store(false, Ordering::Relaxed);

        if let Some(handle) = self._processing_handle.take() {
            handle.abort();
        }

        if let Some(handle) = self._capture_handle.take() {
            let _ = handle.join();
        }
    }

    /// 添加track
    pub fn add_track(&self, uuid: String, track: Arc<TrackLocalStaticSample>) {
        if let Ok(mut tracks) = self.tracks.lock() {
            tracks.insert(
                uuid,
                TrackInfo {
                    track,
                    active: AtomicBool::new(true),
                },
            );
        }
    }

    /// 移除track
    pub fn remove_track(&self, uuid: &str) -> bool {
        if let Ok(mut tracks) = self.tracks.lock() {
            tracks.remove(uuid).is_some()
        } else {
            false
        }
    }

    /// 启用/禁用指定track的写入
    pub fn set_track_active(&self, uuid: &str, active: bool) -> bool {
        if let Ok(tracks) = self.tracks.lock() {
            if let Some(track_info) = tracks.get(uuid) {
                track_info.active.store(active, Ordering::Relaxed);
                return true;
            }
        }
        false
    }

    /// 检查track是否存在且活跃
    pub fn is_track_active(&self, uuid: &str) -> bool {
        if let Ok(tracks) = self.tracks.lock() {
            tracks
                .get(uuid)
                .map(|info| info.active.load(Ordering::Relaxed))
                .unwrap_or(false)
        } else {
            false
        }
    }

    /// 获取活跃track数量
    pub fn active_track_count(&self) -> usize {
        if let Ok(tracks) = self.tracks.lock() {
            tracks
                .values()
                .filter(|info| info.active.load(Ordering::Relaxed))
                .count()
        } else {
            0
        }
    }

    pub fn set_mic_volume(&self, volume: f32) {
        if let Ok(mut mixer) = self.mixer.lock() {
            mixer.set_mic_volume(volume);
        }
    }

    pub fn set_system_volume(&self, volume: f32) {
        if let Ok(mut mixer) = self.mixer.lock() {
            mixer.set_system_volume(volume);
        }
    }

    pub fn set_peer_volume(&self, peer_id: &str, volume: f32) {
        if let Ok(mut volumes) = self.peer_volumes.lock() {
            volumes.insert(peer_id.to_string(), volume.clamp(0.0, 2.0));
        }
    }

    pub fn get_peer_volume(&self, peer_id: &str) -> f32 {
        self.peer_volumes
            .lock()
            .ok()
            .and_then(|v| v.get(peer_id).copied())
            .unwrap_or(1.0)
    }

    pub fn process_peer_audio(&self, peer_id: &str, audio_data: &mut [i16]) {
        let volume = self.get_peer_volume(peer_id);
        for sample in audio_data {
            *sample = (*sample as f32 * volume).clamp(-32768.0, 32767.0) as i16;
        }
    }

    /// 检查捕获是否正在运行
    pub fn is_capture_running(&self) -> bool {
        self.capture_running.load(Ordering::Relaxed)
    }
}

impl Drop for WebRTCAudioSystem {
    fn drop(&mut self) {
        self.stop_capture();
    }
}

// 使用示例
pub async fn start_microphone_audio_stream(
    audio_track: Arc<TrackLocalStaticSample>,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut system = WebRTCAudioSystem::new();

    // 启动捕获
    system.start_capture()?;

    // 添加WebRTC轨道
    let track_uuid = uuid::Uuid::new_v4().to_string();
    system.add_track(track_uuid.clone(), audio_track);

    // 设置音量
    system.set_mic_volume(1.0);
    system.set_system_volume(0.8);
    system.set_peer_volume("peer1", 1.2);

    println!("Audio system started with track: {}", track_uuid);

    // 模拟运行一段时间后暂停某个track
    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
    system.set_track_active(&track_uuid, false);
    println!("Track {} disabled", track_uuid);

    // 再次启用
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    system.set_track_active(&track_uuid, true);
    println!("Track {} enabled", track_uuid);

    // 移除track
    tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
    system.remove_track(&track_uuid);
    println!("Track {} removed", track_uuid);

    // 停止捕获
    system.stop_capture();
    println!("Audio capture stopped");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixer() {
        let mut mixer = AudioMixer::new();
        let result = mixer.mix(&[1000, 2000], &[500, 1000]);
        assert_eq!(result, vec![1500, 3000]);

        mixer.set_mic_volume(0.5);
        let result = mixer.mix(&[1000, 2000], &[500, 1000]);
        assert_eq!(result, vec![1000, 2000]);
    }

    #[test]
    fn test_audio_system() {
        let system = WebRTCAudioSystem::new();

        // 测试track管理
        let track_uuid = "test-uuid";
        assert!(!system.is_track_active(track_uuid));
        assert_eq!(system.active_track_count(), 0);

        // 由于无法创建真实的TrackLocalStaticSample，这里只测试基本逻辑
        assert!(!system.set_track_active(track_uuid, false));
        assert!(!system.remove_track(track_uuid));
    }
}
