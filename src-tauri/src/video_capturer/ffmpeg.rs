use ac_ffmpeg::codec::Encoder;
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, RwLock};
use winapi::shared::ws2ipdef::IN6_PKTINFO;

use ac_ffmpeg::codec::video::frame::{PixelFormat, VideoFrame, VideoFrameMut};
//use ac_ffmpeg::codec::video::{VideoEncoder, VideoFrameMut};
use ac_ffmpeg::time::{TimeBase, Timestamp};
use ac_ffmpeg::Error;
use rusty_duplication::{FrameInfoExt, Scanner, VecCapturer};
use webrtc::track::track_local::track_local_static_sample::TrackLocalStaticSample;

use ac_ffmpeg::codec::video::scaler::{Algorithm, VideoFrameScaler, VideoFrameScalerBuilder};

/// 将 RawFrame（BGRA）转换为目标宽高、YUV420P 像素格式的 VideoFrame
pub fn bgra_to_yuv420p_frame(
    raw_frame: &RawFrame,
    params: &EncodingParams,
    time_base: TimeBase,
    frame_idx: i64,
) -> Result<VideoFrame, Error> {
    // 1. 准备输入 VideoFrameMut：格式 BGRA、尺寸为 raw_frame.width × raw_frame.height
    let input_fmt = PixelFormat::from_str("bgra")
        .map_err(|_| Error::new("Unknown input pixel format 'bgra'".to_string()))?;
    //   .ok_or_else(|| Error::new("Unknown input pixel format 'bgra'".to_string()))?;
    let src_w = raw_frame.width as usize;
    let src_h = raw_frame.height as usize;

    // 分配一个黑色 BGRA 帧，用来拷贝原始数据
    let mut input_video_frame =
        VideoFrameMut::black(input_fmt, src_w, src_h).with_time_base(time_base);

    {
        // 仅有一个平面，打包格式 BGRA，将 raw_frame.data 逐行拷贝到 planes_mut()[0]

        let mut planes = input_video_frame.planes_mut();
        let line_size = planes[0].line_size() as usize;
        let plane_data = planes[0].data_mut();
        let bytes_per_line = raw_frame.width as usize * 4; // BGRA 每像素 4 字节

        for y in 0..src_h {
            let src_offset = y * bytes_per_line;
            let dst_offset = y * line_size;
            // 防止越界
            if src_offset + bytes_per_line <= raw_frame.data.len()
                && dst_offset + bytes_per_line <= plane_data.len()
            {
                plane_data[dst_offset..dst_offset + bytes_per_line]
                    .copy_from_slice(&raw_frame.data[src_offset..src_offset + bytes_per_line]);
            }
        }
    }
    // 冻结输入帧并设置 PTS
    let pts = Timestamp::new(frame_idx, time_base);
    let input_frozen: VideoFrame = input_video_frame.freeze().with_pts(pts);

    // 2. 计算目标宽高（YUV420P 要求偶数对齐）
    let dst_w = if params.width % 2 == 0 {
        params.width
    } else {
        params.width - 1
    };
    let dst_h = if params.height % 2 == 0 {
        params.height
    } else {
        params.height - 1
    };

    // 3. 构建 VideoFrameScaler（注意用 VideoFrameScaler::builder()）
    let output_fmt = PixelFormat::from_str("yuv420p")
        .map_err(|_| Error::new("Unknown input pixel format 'bgra'".to_string()))?;
    //    .ok_or_else(|| Error::new("Unknown output pixel format 'yuv420p'".to_string()))?;
    let mut scaler: VideoFrameScaler = VideoFrameScaler::builder()
        .source_pixel_format(input_fmt)
        .source_width(src_w)
        .source_height(src_h)
        .target_pixel_format(output_fmt)
        .target_width(dst_w as usize)
        .target_height(dst_h as usize)
        // 如果需要指定算法，可以加上 `.algorithm(Algorithm::Bilinear)` 等，默认是 Bicubic
        .algorithm(Algorithm::Bicubic)
        .build()?;

    // 4. 执行缩放与像素格式转换：BGRA → YUV420P
    let mut scaled_frame = scaler.scale(&input_frozen)?;

    // 5. 设置正确的 PTS（scale 后的帧可能保留了输入 PTS，但为了保险，这里重置一次）
    scaled_frame = scaled_frame.with_pts(Timestamp::new(frame_idx, time_base));

    Ok(scaled_frame)
}

#[derive(Clone)]
pub struct RawFrame {
    pub width: u32,
    pub height: u32,
    pub data: Arc<Vec<u8>>, // BGRA格式
    pub timestamp: u64,
    pub frame_id: u64,
}

#[derive(Clone)]
pub struct EncodingParams {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub bitrate: u32,
}

struct EncoderWorker {
    handle: std::thread::JoinHandle<()>, // 改为std::thread::JoinHandle
    shutdown: Arc<AtomicBool>,
}

pub struct MultiStreamManager {
    capture_handle: Option<std::thread::JoinHandle<()>>, // 改为std::thread::JoinHandle
    capture_shutdown: Arc<AtomicBool>,
    frame_broadcast: Option<broadcast::Sender<RawFrame>>,
    encoders: Arc<RwLock<HashMap<String, EncoderWorker>>>,
    frame_counter: Arc<AtomicU64>,
}

impl MultiStreamManager {
    pub fn new() -> Self {
        Self {
            capture_handle: None,
            capture_shutdown: Arc::new(AtomicBool::new(false)),
            frame_broadcast: None,
            encoders: Arc::new(RwLock::new(HashMap::new())),
            frame_counter: Arc::new(AtomicU64::new(0)),
        }
    }

    pub async fn start_capture(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.capture_handle.is_some() {
            return Ok(());
        }

        // 减小缓冲区到最小，避免帧积压
        let (tx, _) = broadcast::channel::<RawFrame>(2); // 从100改为2
        self.frame_broadcast = Some(tx.clone());
        self.capture_shutdown.store(false, Ordering::Relaxed);
        let shutdown_signal = self.capture_shutdown.clone();
        let frame_counter = self.frame_counter.clone();
        let encoders = self.encoders.clone();

        // 使用std::thread::spawn创建独占线程
        let handle = std::thread::spawn(move || {
            // 在独占线程中创建tokio运行时
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                Self::capture_loop(shutdown_signal, frame_counter, tx, encoders).await;
            });
        });

        self.capture_handle = Some(handle);
        Ok(())
    }

    async fn capture_loop(
        shutdown_signal: Arc<AtomicBool>,
        frame_counter: Arc<AtomicU64>,
        tx: broadcast::Sender<RawFrame>,
        encoders: Arc<RwLock<HashMap<String, EncoderWorker>>>,
    ) {
        let mut scanner = match Scanner::new() {
            Ok(s) => s,
            Err(_) => return,
        };

        let monitor = match scanner.next() {
            Some(m) => m,
            None => return,
        };

        let mut capturer: VecCapturer = match monitor.try_into() {
            Ok(c) => c,
            Err(_) => return,
        };

        let mut last_capture = Instant::now();
        let capture_interval = Duration::from_millis(33); // 改为30fps而不是60fps
        let mut skip_counter = 0;

        while !shutdown_signal.load(Ordering::Relaxed) {
            // 检查是否还有编码器在工作
            {
                let encoders_guard = encoders.read().await;
                if encoders_guard.is_empty() {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    continue;
                }
            }

            if let Ok(info) = capturer.capture() {
                if info.desktop_updated() {
                    // 只有满足时间间隔才处理帧
                    if last_capture.elapsed() >= capture_interval {
                        let desc = capturer.monitor().dxgi_outdupl_desc();
                        let frame_id = frame_counter.fetch_add(1, Ordering::Relaxed);

                        let raw_frame = RawFrame {
                            width: desc.ModeDesc.Width,
                            height: desc.ModeDesc.Height,
                            data: Arc::new(capturer.buffer.clone()),
                            timestamp: frame_id,
                            frame_id,
                        };

                        last_capture = Instant::now();

                        // 非阻塞发送，如果通道满了就丢弃旧帧
                        match tx.send(raw_frame) {
                            Ok(_) => {}
                            Err(_) => {
                                // 通道已关闭或满了，跳过这帧
                                skip_counter += 1;
                                if skip_counter % 30 == 0 {
                                    println!(
                                        "Dropped {} frames due to channel congestion",
                                        skip_counter
                                    );
                                }
                            }
                        }
                    } else {
                        // 未到时间间隔，短暂休眠避免CPU占用过高
                        tokio::time::sleep(Duration::from_millis(1)).await;
                    }
                }
            }
        }
    }
    pub async fn add_webrtc_track(
        &self,
        client_uuid: &str,
        track: Arc<TrackLocalStaticSample>,
        params: EncodingParams,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let frame_rx = match &self.frame_broadcast {
            Some(tx) => tx.subscribe(),
            None => return Err("Capture not started".into()),
        };

        let mut encoders = self.encoders.write().await;

        // 如果已存在，先关闭旧的
        if let Some(old_worker) = encoders.remove(client_uuid) {
            old_worker.shutdown.store(true, Ordering::Relaxed);
            // 等待旧线程结束
            if let Err(e) = old_worker.handle.join() {
                eprintln!("Error waiting for old encoder to finish: {:?}", e);
            }
        }
        println!("Adding new encoder for client: {}", client_uuid);
        let shutdown = Arc::new(AtomicBool::new(false));

        // 使用std::thread::spawn创建独占线程
        let handle = std::thread::spawn({
            let shutdown = shutdown.clone();
            move || {
                // 在独占线程中创建tokio运行时
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async {
                    Self::encoder_worker(frame_rx, track, params, shutdown).await;
                });
            }
        });

        let worker = EncoderWorker { handle, shutdown };

        encoders.insert(client_uuid.to_string(), worker);
        Ok(())
    }

    async fn encoder_worker(
        mut frame_rx: broadcast::Receiver<RawFrame>,
        track: Arc<TrackLocalStaticSample>,
        params: EncodingParams,
        shutdown: Arc<AtomicBool>,
    ) {
        let mut encoder = match Self::create_h264_encoder(&params) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("Failed to create encoder: {}", e);
                return;
            }
        };

        let time_base = TimeBase::new(1, params.fps as i32);
        let mut frame_idx = 0i64;
        let mut frames_processed = 0i64;
        let mut frame_sent = 0i64;
        let mut time = Instant::now();
        let frame_duration = Duration::from_millis(1000 / params.fps as u64);

        while !shutdown.load(Ordering::Relaxed) {
            // 减少超时时间，更快响应
            match tokio::time::timeout(Duration::from_millis(50), frame_rx.recv()).await {
                Ok(Ok(mut raw_frame)) => {
                    // 清空接收缓冲区，只处理最新帧
                    loop {
                        match frame_rx.try_recv() {
                            Ok(newer_frame) => {
                                // 有更新的帧，丢弃当前帧
                                raw_frame = newer_frame;
                            }
                            Err(broadcast::error::TryRecvError::Empty) => break,
                            Err(_) => break,
                        }
                    }

                    match bgra_to_yuv420p_frame(&raw_frame, &params, time_base, frame_idx) {
                        Ok(yuv_frame) => {
                            frame_idx += 1;
                            frames_processed += 1;

                            // // 每30帧打印一次状态，确认编码器在工作
                            // if frames_processed % 30 == 0 {
                            //     println!("Processed {} frames", frames_processed);
                            // }

                            if encoder.try_push(yuv_frame).is_ok() {
                                // 立即处理所有可用的包
                                while let Some(packet) = encoder.take().unwrap_or(None) {
                                    let data = packet.data().to_vec();

                                    if let Err(err) = track
                                        .write_sample(&webrtc::media::Sample {
                                            data: data.into(),
                                            duration: frame_duration,
                                            ..Default::default()
                                        })
                                        .await
                                    {
                                        eprintln!("Failed to write sample: {:?}", err);
                                        return;
                                    }
                                }
                                frame_sent += 1;

                                // 每30帧打印一次状态，确认编码器在工作
                                if Instant::now().duration_since(time) >= Duration::from_secs(1) {
                                    println!(
                                        "Processed {} frames,Send {} frames",
                                        frames_processed, frame_sent
                                    );
                                    frame_sent = 0;
                                    frames_processed = 0;
                                    time = Instant::now();
                                }
                            } else {
                                println!("[]fail to push to encoder")
                            }
                        }
                        Err(err) => {
                            eprintln!("Failed to convert frame to YUV420P: {}", err);
                            continue;
                        }
                    }
                }
                Ok(Err(_)) => continue,
                Err(_) => continue, // 超时，继续循环
            }
        }

        // 刷新编码器
        let _ = encoder.try_flush();
        while let Some(packet) = encoder.take().unwrap_or(None) {
            let data = packet.data().to_vec();
            let _ = track
                .write_sample(&webrtc::media::Sample {
                    data: data.into(),
                    duration: frame_duration,
                    ..Default::default()
                })
                .await;
        }
    }

    /// 创建 H.264 编码器（期望输入像素格式 YUV420P、尺寸为 params.width × params.height）
    /// 创建 H.264 编码器，支持GPU加速
    fn create_h264_encoder(
        params: &EncodingParams,
    ) -> Result<ac_ffmpeg::codec::video::VideoEncoder, Error> {
        let time_base = TimeBase::new(1, params.fps as i32);
        let output_pixel_format = PixelFormat::from_str("yuv420p")
            .map_err(|_| Error::new("Unknown pixel format 'yuv420p'".to_string()))?;

        // 按优先级尝试不同编码器
        let encoders = ["libx264", "h264_nvenc", "h264_amf", "h264_qsv"];

        for &encoder_name in &encoders {
            if let Ok(mut builder) = ac_ffmpeg::codec::video::VideoEncoder::builder(encoder_name) {
                println!("Using encoder: {}", encoder_name);

                builder = builder
                    .pixel_format(output_pixel_format)
                    .width(params.width as _)
                    .height(params.height as _)
                    .time_base(time_base)
                    .bit_rate(params.bitrate as _);

                // 根据不同编码器设置不同参数
                match encoder_name {
                    "h264_nvenc" => {
                        // NVIDIA NVENC 参数
                        builder = builder
                            .set_option("preset", "fast") // p1-p7, fast等价于p4
                            .set_option("tune", "ll") // ll = low latency, 不是zerolatency
                            .set_option("profile", "high") // baseline, main, high
                            .set_option("level", "52")
                            .set_option("rc", "cbr") // 码率控制：cbr, vbr, cqp
                            //.set_option("cbr", "1")
                            .set_option("delay", "0") // 0延迟
                            .set_option("zerolatency", "1") // NVENC的零延迟模式
                            .set_option("b_frames", "0") // 禁用B帧
                            .set_option("g", "60")
                            .set_option("refs", "1") // GOP大小（关键帧间隔）
                            .set_option("bufsize", "20000000") // bufsize = 20 Mbps（约=2×码率）
                            .set_option("repeat-headers", "1"); // 每个关键帧重复输出 SPS/PPS
                    }
                    "h264_amf" => {
                        // AMD AMF 参数
                        builder = builder
                            .set_option("usage", "lowlatency") // 低延迟模式
                            .set_option("profile", "256")
                            .set_option("level", "auto")
                            .set_option("rc", "cbr") // 恒定码率
                            .set_option("enforce_hrd", "1") // 强制HRD兼容
                            .set_option("b_frames", "0") // 禁用B帧
                            .set_option("gops_per_idr", "1"); // IDR间隔
                    }
                    "h264_qsv" => {
                        // Intel Quick Sync Video 参数
                        builder = builder
                            .set_option("preset", "faster") // veryfast, faster, fast, medium
                            .set_option("profile", "baseline")
                            .set_option("async_depth", "1") // 异步深度，1为最低延迟
                            .set_option("look_ahead", "0") // 关闭前瞻
                            .set_option("b_strategy", "0") // 禁用B帧策略
                            .set_option("bf", "0") // B帧数量为0
                            .set_option("g", "30"); // GOP大小
                    }
                    "libx264" => {
                        let keyint = std::cmp::min(params.fps * 2, 60); // 最多2秒，最少1秒
                        let min_keyint = params.fps; // 最少1秒一个关键帧
                                                     // CPU x264 参数（原来的设置）
                        builder = builder
                            // .set_option("preset", "ultrafast")
                            // .set_option("tune", "zerolatency")
                            // .set_option("profile", "baseline")
                            // .set_option("intra-refresh", "1")
                            // .set_option("rc-lookahead", "0")
                            // .set_option("sync-lookahead", "0")
                            // .set_option("sliced-threads", "1")
                            // .set_option("b-adapt", "0")
                            // .set_option("bframes", "0")
                            // .set_option("keyint", "30");
                            .set_option("preset", "ultrafast") // 🔥 必须：最快编码速度
                            .set_option("tune", "zerolatency") // 🔥 必须：零延迟调优
                            .set_option("profile", "baseline") // 🔥 必须：baseline profile
                            // 帧结构 - 最简单的配置
                            .set_option("bframes", "0") // 🔥 必须：禁用B帧
                            .set_option("keyint", "25") // 🔥 关键：每25帧一个关键帧(1秒)
                            .set_option("min-keyint", "25") // 🔥 关键：强制关键帧间隔
                            // 延迟控制 - 只保留核心参数
                            .set_option("rc-lookahead", "0") // 🔥 必须：禁用前瞻
                            .set_option("sync-lookahead", "0") // 🔥 必须：禁用同步前瞻
                            // 线程和切片 - 简化设置
                            .set_option("sliced-threads", "1") // 🔥 重要：单线程切片
                            .set_option("threads", "1") // 🔥 关键：单线程编码，避免竞争
                    }
                    _ => {}
                }

                // 尝试构建编码器
                match builder.build() {
                    Ok(encoder) => {
                        println!("Successfully created {} encoder", encoder_name);
                        return Ok(encoder);
                    }
                    Err(e) => {
                        println!(
                            "Failed to create {} encoder: {}, trying next...",
                            encoder_name, e
                        );
                        continue;
                    }
                }
            }
        }

        Err(Error::new("No suitable encoder found".to_string()))
    }

    pub async fn remove_track(
        &mut self,
        client_uuid: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut encoders = self.encoders.write().await;
        if let Some(worker) = encoders.remove(client_uuid) {
            worker.shutdown.store(true, Ordering::Relaxed);
            // 等待线程结束
            if let Err(e) = worker.handle.join() {
                eprintln!("Error waiting for encoder to finish: {:?}", e);
            }
        } else {
            println!("[FFMPEG]关闭{:?}失败", client_uuid)
        }
        if encoders.is_empty() {
            drop(encoders);
            let res = self.stop_capture().await;
            println!("[FFMPEG]无编码器工作，停止录屏");
            return res;
        }
        Ok(())
    }

    pub async fn stop_capture(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // 停止所有编码器
        {
            let mut encoders = self.encoders.write().await;
            for (_, worker) in encoders.drain() {
                worker.shutdown.store(true, Ordering::Relaxed);
                // 等待线程结束
                let _ = worker.handle.join();
            }
        }

        // 停止捕获
        if let Some(handle) = self.capture_handle.take() {
            self.capture_shutdown.store(true, Ordering::Relaxed);
            // 等待线程结束
            let _ = handle.join();
        }

        self.frame_broadcast = None;
        Ok(())
    }
}
