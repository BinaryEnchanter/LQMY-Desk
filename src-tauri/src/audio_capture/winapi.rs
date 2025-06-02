use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc, Mutex,
};
use std::thread;
use std::time::{Duration, Instant};
use tokio::task::JoinHandle;
use webrtc::media::Sample;
use webrtc::track::track_local::track_local_static_sample::TrackLocalStaticSample;
use windows::core::*;
use windows::Win32::Foundation::*;
use windows::Win32::Media::Audio::*;
use windows::Win32::System::Com::*;

const SAMPLE_RATE: u32 = 48000;
const BUFFER_SIZE: usize = 1920; // 20ms * 48000Hz * 2channels / 1000 = 1920 samples
const CHANNELS: usize = 2;

#[derive(Clone)]
pub struct AudioData {
    pub mic: Vec<i16>,
    pub system: Vec<i16>,
    pub timestamp: u64,
}

pub struct AudioMixer {
    mic_volume: f32,
    system_volume: f32,
    // 预分配缓冲区，避免重复分配
    mix_buffer: Vec<i16>,
    byte_buffer: Vec<u8>,
}

impl AudioMixer {
    pub fn new() -> Self {
        Self {
            mic_volume: 1.0,
            system_volume: 1.0,
            mix_buffer: Vec::with_capacity(BUFFER_SIZE),
            byte_buffer: Vec::with_capacity(BUFFER_SIZE * 2),
        }
    }

    pub fn mix_to_bytes(&mut self, mic: &[i16], system: &[i16]) -> &[u8] {
        let len = mic.len().max(system.len());

        // 确保有效的音频长度
        if len == 0 {
            return &[];
        }

        self.mix_buffer.clear();
        self.mix_buffer.reserve(len);

        // 混音处理 - 确保不会溢出
        for i in 0..len {
            let mic_sample = if i < mic.len() {
                (mic[i] as f32 * self.mic_volume).clamp(-32768.0, 32767.0)
            } else {
                0.0
            };

            let sys_sample = if i < system.len() {
                (system[i] as f32 * self.system_volume).clamp(-32768.0, 32767.0)
            } else {
                0.0
            };

            // 混音时防止溢出，使用软限制
            let mixed = mic_sample + sys_sample;
            let limited = if mixed > 32767.0 {
                32767.0
            } else if mixed < -32768.0 {
                -32768.0
            } else {
                mixed
            };

            self.mix_buffer.push(limited as i16);
        }

        // 转换为字节
        self.byte_buffer.clear();
        self.byte_buffer.reserve(self.mix_buffer.len() * 2);

        for &sample in &self.mix_buffer {
            self.byte_buffer.extend_from_slice(&sample.to_le_bytes());
        }

        &self.byte_buffer
    }

    pub fn set_mic_volume(&mut self, vol: f32) {
        self.mic_volume = vol.clamp(0.0, 2.0);
    }
    pub fn set_system_volume(&mut self, vol: f32) {
        self.system_volume = vol.clamp(0.0, 2.0);
    }
}

// 优化的音频捕获线程 - 更高频率，更低延迟
fn audio_capture_thread(
    sender: mpsc::Sender<AudioData>,
    running: Arc<AtomicBool>,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    unsafe {
        CoInitializeEx(None, COINIT_MULTITHREADED).ok();

        let enumerator: IMMDeviceEnumerator =
            CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)?;

        // 麦克风 - 使用更小的缓冲区
        let mic_device = enumerator.GetDefaultAudioEndpoint(eCapture, eConsole)?;
        let mic_client: IAudioClient = mic_device.Activate(CLSCTX_ALL, None)?;
        let mic_format = mic_client.GetMixFormat()?;
        // 减少缓冲区大小以降低延迟，但确保足够的缓冲
        mic_client.Initialize(AUDCLNT_SHAREMODE_SHARED, 0, 1000000, 0, mic_format, None)?; // 100ms buffer
        let mic_capture: IAudioCaptureClient = mic_client.GetService()?;

        // 系统音频
        let sys_device = enumerator.GetDefaultAudioEndpoint(eRender, eConsole)?;
        let sys_client: IAudioClient = sys_device.Activate(CLSCTX_ALL, None)?;
        let sys_format = sys_client.GetMixFormat()?;
        sys_client.Initialize(
            AUDCLNT_SHAREMODE_SHARED,
            AUDCLNT_STREAMFLAGS_LOOPBACK,
            1000000,
            0,
            sys_format,
            None,
        )?;
        let sys_capture: IAudioCaptureClient = sys_client.GetService()?;

        mic_client.Start()?;
        sys_client.Start()?;

        let mut timestamp = 0u64;
        let mut last_send = Instant::now();

        while running.load(Ordering::Relaxed) {
            // 稳定的20ms间隔，确保音频连续性
            thread::sleep(Duration::from_millis(20));

            let mut mic_data = Vec::new();
            let mut sys_data = Vec::new();

            // 捕获麦克风 - 循环获取所有可用数据
            loop {
                let packet_len = match mic_capture.GetNextPacketSize() {
                    Ok(len) => len,
                    Err(_) => break,
                };

                if packet_len == 0 {
                    break;
                }

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
                } else {
                    break;
                }
            }

            // 捕获系统音频 - 循环获取所有可用数据
            loop {
                let packet_len = match sys_capture.GetNextPacketSize() {
                    Ok(len) => len,
                    Err(_) => break,
                };

                if packet_len == 0 {
                    break;
                }

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
                } else {
                    break;
                }
            }

            // 始终发送数据，即使是空的（保持时序）
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

        mic_client.Stop().ok();
        sys_client.Stop().ok();
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
    capture_handle: Arc<Mutex<Option<thread::JoinHandle<()>>>>,
    processing_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
}

impl WebRTCAudioSystem {
    pub fn new() -> Self {
        let mixer = Arc::new(Mutex::new(AudioMixer::new()));
        let tracks = Arc::new(Mutex::new(HashMap::new()));
        let peer_volumes = Arc::new(Mutex::new(HashMap::new()));
        let capture_running = Arc::new(AtomicBool::new(false));
        let processing_running = Arc::new(AtomicBool::new(false));
        let capture_handle = Arc::new(Mutex::new(None));
        let processing_handle = Arc::new(Mutex::new(None));
        Self {
            mixer,
            tracks,
            peer_volumes,
            capture_running,
            processing_running,
            capture_handle,
            processing_handle,
        }
    }

    pub fn start_capture(&self) -> std::result::Result<(), Box<dyn std::error::Error>> {
        if self.capture_running.load(Ordering::Relaxed) {
            return Ok(());
        }

        let (tx, rx) = mpsc::channel::<AudioData>();

        let capture_running = Arc::clone(&self.capture_running);
        capture_running.store(true, Ordering::Relaxed);

        let capture_running_clone = Arc::clone(&capture_running);
        let capture_handle = thread::spawn(move || {
            if let Err(e) = audio_capture_thread(tx, capture_running_clone) {
                eprintln!("Audio capture error: {}", e);
            }
        });

        let processing_running = Arc::clone(&self.processing_running);
        processing_running.store(true, Ordering::Relaxed);

        let mixer_clone = Arc::clone(&self.mixer);
        let tracks_clone = Arc::clone(&self.tracks);
        let processing_running_clone = Arc::clone(&processing_running);

        // 使用标准的tokio spawn，避免复杂的运行时嵌套
        let processing_handle = tokio::spawn(async move {
            while processing_running_clone.load(Ordering::Relaxed) {
                // 使用阻塞接收，确保音频数据的连续性
                match rx.recv() {
                    Ok(audio) => {
                        // 快速处理音频数据
                        let (sample_data, active_tracks) = {
                            let sample_bytes = if let Ok(mut mixer) = mixer_clone.lock() {
                                mixer.mix_to_bytes(&audio.mic, &audio.system).to_vec()
                            } else {
                                continue;
                            };

                            let tracks = if let Ok(tracks_guard) = tracks_clone.lock() {
                                tracks_guard
                                    .values()
                                    .filter(|info| info.active.load(Ordering::Relaxed))
                                    .map(|info| Arc::clone(&info.track))
                                    .collect::<Vec<_>>()
                            } else {
                                continue;
                            };

                            (sample_bytes, tracks)
                        };

                        // 只有在有有效数据时才发送
                        if !sample_data.is_empty() && !active_tracks.is_empty() {
                            let sample = Sample {
                                data: sample_data.into(),
                                duration: Duration::from_millis(20),
                                ..Default::default()
                            };

                            let tasks: Vec<_> = active_tracks
                                .into_iter()
                                .map(|track| {
                                    let sample_clone = clone_sample(&sample);
                                    async move { track.write_sample(&sample_clone).await }
                                })
                                .collect();

                            let _ = futures::future::join_all(tasks).await;
                        }
                    }
                    Err(_) => break, // 通道关闭
                }
            }
        });

        // 通过Arc<Mutex<>>来存储句柄
        if let Ok(mut handle_guard) = self.capture_handle.lock() {
            *handle_guard = Some(capture_handle);
        }

        if let Ok(mut handle_guard) = self.processing_handle.lock() {
            *handle_guard = Some(processing_handle);
        }

        Ok(())
    }

    pub fn stop_capture(&self) {
        self.capture_running.store(false, Ordering::Relaxed);
        self.processing_running.store(false, Ordering::Relaxed);

        // 通过Arc<Mutex<>>来访问和修改句柄
        if let Ok(mut handle_guard) = self.processing_handle.lock() {
            if let Some(handle) = handle_guard.take() {
                handle.abort();
            }
        }

        if let Ok(mut handle_guard) = self.capture_handle.lock() {
            if let Some(handle) = handle_guard.take() {
                let _ = handle.join();
            }
        }
    }

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

    pub fn remove_track(&self, uuid: &str) -> bool {
        if let Ok(mut tracks) = self.tracks.lock() {
            tracks.remove(uuid).is_some()
        } else {
            false
        }
    }

    pub fn set_track_active(&self, uuid: &str, active: bool) -> bool {
        if let Ok(tracks) = self.tracks.lock() {
            if let Some(track_info) = tracks.get(uuid) {
                track_info.active.store(active, Ordering::Relaxed);
                return true;
            }
        }
        false
    }

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

    pub fn is_capture_running(&self) -> bool {
        self.capture_running.load(Ordering::Relaxed)
    }
}

impl Drop for WebRTCAudioSystem {
    fn drop(&mut self) {
        self.stop_capture();
    }
}
fn clone_sample(sample: &Sample) -> Sample {
    Sample {
        data: sample.data.clone(),
        timestamp: sample.timestamp.clone(),
        duration: sample.duration.clone(),
        packet_timestamp: sample.packet_timestamp.clone(),
        prev_dropped_packets: sample.prev_dropped_packets.clone(),
        prev_padding_packets: sample.prev_padding_packets.clone(),
    }
}
