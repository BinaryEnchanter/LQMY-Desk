// use ac_ffmpeg::codec::Encoder;
// use std::collections::HashMap;
// use std::str::FromStr;
// use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
// use std::sync::Arc;
// use std::time::{Duration, Instant};
// use tokio::sync::{broadcast, mpsc, RwLock};
// use tokio::task::JoinHandle;

// use ac_ffmpeg::codec::video::frame::{PixelFormat, VideoFrame, VideoFrameMut};
// //use ac_ffmpeg::codec::video::{VideoEncoder, VideoFrameMut};
// use ac_ffmpeg::time::{TimeBase, Timestamp};
// use ac_ffmpeg::Error;
// use rusty_duplication::{FrameInfoExt, Scanner, VecCapturer};
// use webrtc::track::track_local::track_local_static_sample::TrackLocalStaticSample;

// use ac_ffmpeg::codec::video::scaler::{Algorithm, VideoFrameScaler, VideoFrameScalerBuilder};

// /// 将 RawFrame（BGRA）转换为目标宽高、YUV420P 像素格式的 VideoFrame
// pub fn bgra_to_yuv420p_frame(
//     raw_frame: &RawFrame,
//     params: &EncodingParams,
//     time_base: TimeBase,
//     frame_idx: i64,
// ) -> Result<VideoFrame, Error> {
//     // 1. 准备输入 VideoFrameMut：格式 BGRA、尺寸为 raw_frame.width × raw_frame.height
//     let input_fmt = PixelFormat::from_str("bgra")
//         .map_err(|_| Error::new("Unknown input pixel format 'bgra'".to_string()))?;
//     //   .ok_or_else(|| Error::new("Unknown input pixel format 'bgra'".to_string()))?;
//     let src_w = raw_frame.width as usize;
//     let src_h = raw_frame.height as usize;

//     // 分配一个黑色 BGRA 帧，用来拷贝原始数据
//     let mut input_video_frame =
//         VideoFrameMut::black(input_fmt, src_w, src_h).with_time_base(time_base);

//     {
//         // 仅有一个平面，打包格式 BGRA，将 raw_frame.data 逐行拷贝到 planes_mut()[0]

//         let mut planes = input_video_frame.planes_mut();
//         let line_size = planes[0].line_size() as usize;
//         let plane_data = planes[0].data_mut();
//         let bytes_per_line = raw_frame.width as usize * 4; // BGRA 每像素 4 字节

//         for y in 0..src_h {
//             let src_offset = y * bytes_per_line;
//             let dst_offset = y * line_size;
//             // 防止越界
//             if src_offset + bytes_per_line <= raw_frame.data.len()
//                 && dst_offset + bytes_per_line <= plane_data.len()
//             {
//                 plane_data[dst_offset..dst_offset + bytes_per_line]
//                     .copy_from_slice(&raw_frame.data[src_offset..src_offset + bytes_per_line]);
//             }
//         }
//     }
//     // 冻结输入帧并设置 PTS
//     let pts = Timestamp::new(frame_idx, time_base);
//     let input_frozen: VideoFrame = input_video_frame.freeze().with_pts(pts);

//     // 2. 计算目标宽高（YUV420P 要求偶数对齐）
//     let dst_w = if params.width % 2 == 0 {
//         params.width
//     } else {
//         params.width - 1
//     };
//     let dst_h = if params.height % 2 == 0 {
//         params.height
//     } else {
//         params.height - 1
//     };

//     // 3. 构建 VideoFrameScaler（注意用 VideoFrameScaler::builder()）
//     let output_fmt = PixelFormat::from_str("yuv420p")
//         .map_err(|_| Error::new("Unknown input pixel format 'bgra'".to_string()))?;
//     //    .ok_or_else(|| Error::new("Unknown output pixel format 'yuv420p'".to_string()))?;
//     let mut scaler: VideoFrameScaler = VideoFrameScaler::builder()
//         .source_pixel_format(input_fmt)
//         .source_width(src_w)
//         .source_height(src_h)
//         .target_pixel_format(output_fmt)
//         .target_width(dst_w as usize)
//         .target_height(dst_h as usize)
//         // 如果需要指定算法，可以加上 `.algorithm(Algorithm::Bilinear)` 等，默认是 Bicubic
//         .algorithm(Algorithm::Bicubic)
//         .build()?;

//     // 4. 执行缩放与像素格式转换：BGRA → YUV420P
//     let mut scaled_frame = scaler.scale(&input_frozen)?;

//     // 5. 设置正确的 PTS（scale 后的帧可能保留了输入 PTS，但为了保险，这里重置一次）
//     scaled_frame = scaled_frame.with_pts(Timestamp::new(frame_idx, time_base));

//     Ok(scaled_frame)
// }

// #[derive(Clone)]
// pub struct RawFrame {
//     pub width: u32,
//     pub height: u32,
//     pub data: Arc<Vec<u8>>, // BGRA格式
//     pub timestamp: u64,
//     pub frame_id: u64,
// }

// #[derive(Clone)]
// pub struct EncodingParams {
//     pub width: u32,
//     pub height: u32,
//     pub fps: u32,
//     pub bitrate: u32,
// }

// struct EncoderWorker {
//     handle: JoinHandle<()>,
//     shutdown: Arc<AtomicBool>,
// }

// pub struct MultiStreamManager {
//     capture_handle: Option<JoinHandle<()>>,
//     capture_shutdown: Arc<AtomicBool>,
//     frame_broadcast: Option<broadcast::Sender<RawFrame>>,
//     encoders: Arc<RwLock<HashMap<String, EncoderWorker>>>,
//     frame_counter: Arc<AtomicU64>,
// }

// impl MultiStreamManager {
//     pub fn new() -> Self {
//         Self {
//             capture_handle: None,
//             capture_shutdown: Arc::new(AtomicBool::new(false)),
//             frame_broadcast: None,
//             encoders: Arc::new(RwLock::new(HashMap::new())),
//             frame_counter: Arc::new(AtomicU64::new(0)),
//         }
//     }

//     pub async fn start_capture(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//         if self.capture_handle.is_some() {
//             return Ok(());
//         }

//         let (tx, _) = broadcast::channel::<RawFrame>(100);
//         self.frame_broadcast = Some(tx.clone());
//         self.capture_shutdown.store(false, Ordering::Relaxed);
//         let shutdown_signal = self.capture_shutdown.clone();
//         let frame_counter = self.frame_counter.clone();
//         let encoders = self.encoders.clone();

//         let handle = tokio::spawn(async move {
//             Self::capture_loop(shutdown_signal, frame_counter, tx, encoders).await;
//         });

//         self.capture_handle = Some(handle);
//         Ok(())
//     }

//     async fn capture_loop(
//         shutdown_signal: Arc<AtomicBool>,
//         frame_counter: Arc<AtomicU64>,
//         tx: broadcast::Sender<RawFrame>,
//         encoders: Arc<RwLock<HashMap<String, EncoderWorker>>>,
//     ) {
//         let mut scanner = match Scanner::new() {
//             Ok(s) => s,
//             Err(_) => return,
//         };

//         let monitor = match scanner.next() {
//             Some(m) => m,
//             None => return,
//         };

//         let mut capturer: VecCapturer = match monitor.try_into() {
//             Ok(c) => c,
//             Err(_) => return,
//         };

//         let mut last_capture = Instant::now();
//         let capture_interval = Duration::from_millis(16); // ~60fps

//         while !shutdown_signal.load(Ordering::Relaxed) {
//             if last_capture.elapsed() < capture_interval {
//                 tokio::time::sleep(Duration::from_millis(1)).await;
//                 continue;
//             }

//             // 检查是否还有编码器在工作
//             {
//                 let encoders_guard = encoders.read().await;
//                 if encoders_guard.is_empty() {
//                     tokio::time::sleep(Duration::from_millis(10)).await;
//                     continue;
//                 }
//             }

//             if let Ok(info) = capturer.capture() {
//                 if info.desktop_updated() {
//                     let desc = capturer.monitor().dxgi_outdupl_desc();
//                     let frame_id = frame_counter.fetch_add(1, Ordering::Relaxed);

//                     let raw_frame = RawFrame {
//                         width: desc.ModeDesc.Width,
//                         height: desc.ModeDesc.Height,
//                         data: Arc::new(capturer.buffer.clone()),
//                         timestamp: frame_id,
//                         frame_id,
//                     };
//                     //println!("[RAW]{:?}*{:?}", raw_frame.width, raw_frame.height);
//                     last_capture = Instant::now();

//                     // 广播BGRA帧给所有编码器
//                     let _ = tx.send(raw_frame);
//                 }
//             }
//         }
//     }

//     pub async fn add_webrtc_track(
//         &self,
//         client_uuid: &str,
//         track: Arc<TrackLocalStaticSample>,
//         params: EncodingParams,
//     ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//         let frame_rx = match &self.frame_broadcast {
//             Some(tx) => tx.subscribe(),
//             None => return Err("Capture not started".into()),
//         };

//         let mut encoders = self.encoders.write().await;

//         // 如果已存在，先关闭旧的
//         if let Some(old_worker) = encoders.remove(client_uuid) {
//             old_worker.shutdown.store(true, Ordering::Relaxed);
//             old_worker.handle.abort();
//         }

//         let shutdown = Arc::new(AtomicBool::new(false));

//         let handle = tokio::spawn(Self::encoder_worker(
//             frame_rx,
//             track,
//             params,
//             shutdown.clone(),
//         ));

//         let worker = EncoderWorker { handle, shutdown };

//         encoders.insert(client_uuid.to_string(), worker);
//         Ok(())
//     }

//     async fn encoder_worker(
//         mut frame_rx: broadcast::Receiver<RawFrame>,
//         track: Arc<TrackLocalStaticSample>,
//         params: EncodingParams,
//         shutdown: Arc<AtomicBool>,
//     ) {
//         let mut encoder = match Self::create_h264_encoder(&params) {
//             Ok(e) => e,
//             Err(e) => {
//                 eprintln!("Failed to create encoder: {}", e);
//                 return;
//             }
//         };

//         let time_base = TimeBase::new(1, params.fps as i32);
//         let mut frame_idx = 0i64;

//         while !shutdown.load(Ordering::Relaxed) {
//             match tokio::time::timeout(Duration::from_millis(100), frame_rx.recv()).await {
//                 Ok(Ok(raw_frame)) => {
//                     match bgra_to_yuv420p_frame(&raw_frame, &params, time_base, frame_idx) {
//                         Ok(yuv_frame) => {
//                             // 转换成功，更新帧索引
//                             frame_idx += 1;

//                             // 如果你想打印帧信息，可以加上：
//                             // println!(
//                             //     "[ENCODER] YUVFRAME: {}×{}  (PTS={})",
//                             //     yuv_frame..width(),
//                             //     yuv_frame.height(),
//                             //     yuv_frame.pts().unwrap_or_default()
//                             // );

//                             // —— 4. 将 YUV420P 的 video_frame 推给编码器
//                             if encoder.try_push(yuv_frame).is_ok() {
//                                 // —— 5. 从编码器读取所有已经 ready 的 packet，写到 WebRTC Track
//                                 while let Some(packet) = encoder.take().unwrap_or(None) {
//                                     let data = packet.data().to_vec();
//                                     //let is_keyframe = packet.is_key();
//                                     if let Err(err) = track
//                                         .write_sample(&webrtc::media::Sample {
//                                             data: data.into(),
//                                             duration: Duration::from_millis(
//                                                 1000 / params.fps as u64,
//                                             ),

//                                             ..Default::default()
//                                         })
//                                         .await
//                                     {
//                                         eprintln!("Failed to write sample: {:?}", err);
//                                         return;
//                                     }
//                                 }
//                             }
//                         }
//                         Err(err) => {
//                             // 转换失败，跳过这帧
//                             eprintln!("Failed to convert frame to YUV420P: {}", err);
//                             continue;
//                         }
//                     }
//                 }

//                 Ok(Err(_)) => continue,
//                 Err(_) => continue,
//             }
//         }

//         // 刷新编码器
//         let _ = encoder.try_flush();
//         while let Some(packet) = encoder.take().unwrap_or(None) {
//             let data = packet.data().to_vec();
//             let _ = track
//                 .write_sample(&webrtc::media::Sample {
//                     data: data.into(),
//                     duration: Duration::from_millis(1000 / params.fps as u64),
//                     ..Default::default()
//                 })
//                 .await;
//         }
//     }

//     /// 创建 H.264 编码器（期望输入像素格式 YUV420P、尺寸为 params.width × params.height）
//     fn create_h264_encoder(
//         params: &EncodingParams,
//     ) -> Result<ac_ffmpeg::codec::video::VideoEncoder, Error> {
//         let time_base = TimeBase::new(1, params.fps as i32);
//         let output_pixel_format = PixelFormat::from_str("yuv420p")
//             .map_err(|_| Error::new("Unknown input pixel format 'bgra'".to_string()))?;
//         //    .ok_or_else(|| Error::new("Unknown pixel format 'yuv420p'".to_string()))?;

//         ac_ffmpeg::codec::video::VideoEncoder::builder("libx264")?
//             .pixel_format(output_pixel_format)
//             .width(params.width as _)
//             .height(params.height as _)
//             .time_base(time_base)
//             .bit_rate(params.bitrate as _)
//             .set_option("preset", "ultrafast")
//             .set_option("tune", "zerolatency")
//             .set_option("profile", "baseline")
//             //.set_option("keyint", (params.fps * 2).to_string().as_str())
//             .set_option("min-keyint", params.fps.to_string().as_str())
//             // 提高容错性
//             .set_option("x264-params", "keyint=30:scenecut=0:repeat-headers=1")
//             .set_option("crf", "23")
//             .build()
//     }

//     pub async fn remove_track(
//         &mut self,
//         client_uuid: &str,
//     ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//         let mut encoders = self.encoders.write().await;
//         if let Some(worker) = encoders.remove(client_uuid) {
//             worker.shutdown.store(true, Ordering::Relaxed);
//             worker.handle.abort();
//         } else {
//             println!("[FFMPEG]关闭{:?}失败", client_uuid)
//         }
//         if encoders.is_empty() {
//             drop(encoders);
//             let res = self.stop_capture().await;
//             println!("[FFMPEG]无编码器工作，停止录屏");
//             return res;
//         }
//         Ok(())
//     }

//     pub async fn stop_capture(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//         // 停止所有编码器
//         {
//             let mut encoders = self.encoders.write().await;
//             for (_, worker) in encoders.drain() {
//                 worker.shutdown.store(true, Ordering::Relaxed);
//                 worker.handle.abort();
//             }
//         }

//         // 停止捕获
//         if let Some(handle) = self.capture_handle.take() {
//             self.capture_shutdown.store(true, Ordering::Relaxed);
//             handle.abort();
//         }

//         self.frame_broadcast = None;
//         Ok(())
//     }
// }

use ac_ffmpeg::codec::Encoder;
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, mpsc, RwLock};
use tokio::task::JoinHandle;

use ac_ffmpeg::codec::video::frame::{PixelFormat, VideoFrame, VideoFrameMut};
use ac_ffmpeg::time::{TimeBase, Timestamp};
use ac_ffmpeg::Error;
use rusty_duplication::{FrameInfoExt, Scanner, VecCapturer};
use webrtc::track::track_local::track_local_static_sample::TrackLocalStaticSample;

use ac_ffmpeg::codec::video::scaler::{Algorithm, VideoFrameScaler, VideoFrameScalerBuilder};

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

/// 预先创建一次、可复用的缩放器封装
struct FrameConverter {
    scaler: VideoFrameScaler,
    src_width: usize,
    src_height: usize,
    src_format: PixelFormat,
    dst_width: usize,
    dst_height: usize,
    dst_format: PixelFormat,
}

impl FrameConverter {
    /// 在 EncoderWorker 初始化时调用，只创建一次
    pub fn new(src_w: u32, src_h: u32, params: &EncodingParams) -> Result<Self, Error> {
        // 源格式：BGRA
        let src_format = PixelFormat::from_str("bgra")
            .map_err(|_| Error::new("Unknown pixel format 'bgra'".to_string()))?;
        // 目标格式：YUV420P
        let dst_format = PixelFormat::from_str("yuv420p")
            .map_err(|_| Error::new("Unknown pixel format 'yuv420p'".to_string()))?;

        // 确保宽高为偶数
        let dst_w = if params.width % 2 == 0 {
            params.width as usize
        } else {
            (params.width - 1) as usize
        };
        let dst_h = if params.height % 2 == 0 {
            params.height as usize
        } else {
            (params.height - 1) as usize
        };

        let scaler = VideoFrameScaler::builder()
            .source_pixel_format(src_format)
            .source_width(src_w as usize)
            .source_height(src_h as usize)
            .target_pixel_format(dst_format)
            .target_width(dst_w)
            .target_height(dst_h)
            .algorithm(Algorithm::Bicubic) // 或者用你需要的算法
            .build()?;

        Ok(FrameConverter {
            scaler,
            src_width: src_w as usize,
            src_height: src_h as usize,
            src_format,
            dst_width: dst_w,
            dst_height: dst_h,
            dst_format,
        })
    }

    ///「零拷贝」或一次性 memcpy 将 RawFrame 的 BGRA 数据放入 VideoFrameMut，然后缩放到 YUV420P
    pub fn convert(
        &mut self,
        raw_frame: &RawFrame,
        time_base: TimeBase,
        frame_idx: i64,
    ) -> Result<VideoFrame, Error> {
        // 1. 构造 VideoFrameMut，直接用 raw_frame.data 的内存
        //    如果 ac_ffmpeg 支持 unsafe 的 from_raw_buffer 方法，就可以避免 memcpy。这里假设只能 memcpy:
        let mut input_frame =
            VideoFrameMut::black(self.src_format, self.src_width, self.src_height)
                .with_time_base(time_base);

        {
            let line_size = input_frame.planes()[0].line_size() as usize;
            let mut plane0 = input_frame.planes_mut();
            let data_mut = plane0[0].data_mut();
            //let line_size = plane0[0].line_size() as usize;
            let bytes_per_line = self.src_width * 4; // BGRA 每像素4字节

            // 做一次性拷贝：如果行对齐方式是连续的，可以 memcpy 整块
            // 下面用单行 memcpy 确保不会越界
            for y in 0..self.src_height {
                let src_offset = y * bytes_per_line;
                let dst_offset = y * line_size;
                if src_offset + bytes_per_line <= raw_frame.data.len()
                    && dst_offset + bytes_per_line <= data_mut.len()
                {
                    // 一次性 memcpy
                    data_mut[dst_offset..dst_offset + bytes_per_line]
                        .copy_from_slice(&raw_frame.data[src_offset..src_offset + bytes_per_line]);
                }
            }
        }
        // 冻结并设置 PTS
        let pts = Timestamp::new(frame_idx, time_base);
        let input_frozen: VideoFrame = input_frame.freeze().with_pts(pts);

        // 2. 缩放并转换格式
        let mut scaled = self.scaler.scale(&input_frozen)?;
        // 3. 重新设置 PTS（以防 scaler 丢失）
        scaled = scaled.with_pts(Timestamp::new(frame_idx, time_base));
        Ok(scaled)
    }
}

struct EncoderWorker {
    handle: JoinHandle<()>,
    shutdown: Arc<AtomicBool>,
}

/// 主管理结构体
pub struct MultiStreamManager {
    capture_handle: Option<JoinHandle<()>>,
    capture_shutdown: Arc<AtomicBool>,
    frame_broadcast: Option<broadcast::Sender<RawFrame>>,
    /// 只在 add/remove track 时需要写锁。平时通过 atomic_count 判断是否有活跃编码器
    encoders: Arc<RwLock<HashMap<String, EncoderWorker>>>,
    /// 记录当前活跃编码器数量
    active_count: Arc<AtomicUsize>,
    /// 全局帧计数
    frame_counter: Arc<AtomicU64>,
}

impl MultiStreamManager {
    pub fn new() -> Self {
        Self {
            capture_handle: None,
            capture_shutdown: Arc::new(AtomicBool::new(false)),
            frame_broadcast: None,
            encoders: Arc::new(RwLock::new(HashMap::new())),
            active_count: Arc::new(AtomicUsize::new(0)),
            frame_counter: Arc::new(AtomicU64::new(0)),
        }
    }

    /// 启动桌面捕获循环
    pub async fn start_capture(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.capture_handle.is_some() {
            return Ok(());
        }
        let (tx, _) = broadcast::channel::<RawFrame>(10); // 缓冲 10 帧就够
        self.frame_broadcast = Some(tx.clone());
        self.capture_shutdown.store(false, Ordering::Relaxed);

        let shutdown_signal = self.capture_shutdown.clone();
        let frame_counter = self.frame_counter.clone();
        let encoders = self.encoders.clone();
        let active_count = self.active_count.clone();

        let handle = tokio::spawn(async move {
            // 初始化 screen capturer
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
            let capture_interval = Duration::from_millis(16); // ~60 fps

            while !shutdown_signal.load(Ordering::Relaxed) {
                // 如果当前没有活跃编码器，就休眠一段时间，避免无谓捕获
                if active_count.load(Ordering::Relaxed) == 0 {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    continue;
                }

                // 控制帧率
                if last_capture.elapsed() < capture_interval {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                    continue;
                }

                if let Ok(info) = capturer.capture() {
                    if info.desktop_updated() {
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
                        // 广播给所有编码器
                        let _ = tx.send(raw_frame);
                    }
                }
            }
        });

        self.capture_handle = Some(handle);
        Ok(())
    }

    /// 将 WebRTC Track 加入到多路编码管理中
    pub async fn add_webrtc_track(
        &self,
        client_uuid: &str,
        track: Arc<TrackLocalStaticSample>,
        params: EncodingParams,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut frame_rx = match &self.frame_broadcast {
            Some(tx) => tx.subscribe(),
            None => return Err("Capture not started".into()),
        };

        let mut enc_map = self.encoders.write().await;

        // 如果已存在，则先下线旧的
        if let Some(old_worker) = enc_map.remove(client_uuid) {
            old_worker.shutdown.store(true, Ordering::Relaxed);
            // 等待旧任务结束（可加 timeout）
            let _ = old_worker.handle.await;
        }

        // 标记有新编码器加入
        self.active_count.fetch_add(1, Ordering::Relaxed);

        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown1 = shutdown.clone();
        let active_count = self.active_count.clone();

        // 在启动时：检测第一帧的宽高，用于 FrameConverter
        // 这里简化，假设屏幕分辨率恒定。也可以在接收到第一帧 raw_frame 后再创建 FrameConverter
        let dummy_raw = {
            // 订阅一帧用作获取分辨率。实际第一个可用 raw_frame 里拿到的宽高一样。
            match frame_rx.recv().await {
                Ok(f) => f,
                Err(_) => return Err("Failed to receive initial frame".into()),
            }
        };
        let src_w = dummy_raw.width;
        let src_h = dummy_raw.height;

        // 重建订阅器：因为上面我们已经消费了一帧，需要重新订阅
        let mut frame_rx = self.frame_broadcast.as_ref().unwrap().subscribe();

        // 创建 FrameConverter，只做一次
        let mut converter = FrameConverter::new(src_w, src_h, &params)?;

        // 当前时间基
        let time_base = TimeBase::new(1, params.fps as i32);

        // 如果需要把第一帧也送编码，可以在此处再调用一次 converter.convert(dummy_raw,..) 并 push
        // 这部分看业务是否需要。为了简化，此处不重复推送第一帧。

        // 启动编码任务
        let handle = tokio::spawn(async move {
            // 创建 x264 编码器
            let mut encoder = match {
                let mut builder =
                    ac_ffmpeg::codec::video::VideoEncoder::builder("libx264").unwrap();
                builder
                    .pixel_format(PixelFormat::from_str("yuv420p").unwrap())
                    .width(params.width as _)
                    .height(params.height as _)
                    .time_base(time_base)
                    .bit_rate(params.bitrate as _)
                    // ---- 优化点：指定线程数量，利用多核 ----
                    .set_option("threads", "4")
                    .set_option("preset", "ultrafast")
                    .set_option("tune", "zerolatency")
                    .set_option("profile", "baseline")
                    .set_option("min-keyint", params.fps.to_string().as_str())
                    .set_option("x264-params", "keyint=30:scenecut=0:repeat-headers=1")
                    .set_option("crf", "23")
                    .build()
                //builder.build()
            } {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("Failed to create encoder: {}", e);
                    // 任务结束前把 active_count--，避免漏减
                    active_count.fetch_sub(1, Ordering::Relaxed);
                    return;
                }
            };

            let mut frame_idx = 0i64;
            // 编码循环：直接阻塞在 recv().await
            loop {
                // 先检查是否需要退出
                if shutdown.load(Ordering::Relaxed) {
                    break;
                }
                // 等待新帧
                let raw_frame = match frame_rx.recv().await {
                    Ok(f) => f,
                    Err(_) => {
                        // 频道关闭或错误，直接退出
                        break;
                    }
                };

                // 转换并推入编码器
                match converter.convert(&raw_frame, time_base, frame_idx) {
                    Ok(yuv_frame) => {
                        frame_idx += 1;
                        if encoder.try_push(yuv_frame).is_ok() {
                            // 读取所有可用 packet 并写给 WebRTC
                            while let Some(packet) = encoder.take().unwrap_or(None) {
                                let data = packet.data().to_vec();
                                let sample = webrtc::media::Sample {
                                    data: data.into(),
                                    duration: Duration::from_millis(1000 / params.fps as u64),
                                    ..Default::default()
                                };
                                if let Err(err) = track.write_sample(&sample).await {
                                    eprintln!("Failed to write sample: {:?}", err);
                                    // 出错则退出循环
                                    shutdown.store(true, Ordering::Relaxed);
                                    break;
                                }
                            }
                        }
                    }
                    Err(err) => {
                        eprintln!("Convert frame error: {}", err);
                        // 本帧跳过，不退出
                        continue;
                    }
                }
            }

            // 任务退出前：flush 编码器剩余数据
            let _ = encoder.try_flush();
            while let Some(packet) = encoder.take().unwrap_or(None) {
                let data = packet.data().to_vec();
                let sample = webrtc::media::Sample {
                    data: data.into(),
                    duration: Duration::from_millis(1000 / params.fps as u64),
                    ..Default::default()
                };
                let _ = track.write_sample(&sample).await;
            }

            // 任务真正结束时，把 active_count--，表示一个编码器退出了
            active_count.fetch_sub(1, Ordering::Relaxed);
        });

        // 存进 map
        let worker = EncoderWorker {
            handle: handle,
            shutdown: shutdown1,
        };
        enc_map.insert(client_uuid.to_string(), worker);
        Ok(())
    }

    /// 移除某一路编码 track
    pub async fn remove_track(
        &mut self,
        client_uuid: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut enc_map = self.encoders.write().await;
        if let Some(worker) = enc_map.remove(client_uuid) {
            // 优雅通知退出
            worker.shutdown.store(true, Ordering::Relaxed);
            // 等待结束
            let _ = worker.handle.await;
        } else {
            println!("[FFMPEG] 未找到要关闭的编码器: {:?}", client_uuid);
        }

        // 如果再也没有活跃编码器，就停止捕获
        if self.active_count.load(Ordering::Relaxed) == 0 {
            drop(enc_map);
            let _ = self.stop_capture().await;
            println!("[FFMPEG] 无编码器，停止捕获。");
        }
        Ok(())
    }

    /// 停止全部捕获和编码
    pub async fn stop_capture(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // 通知所有 encoder 退出并等待
        {
            let mut enc_map = self.encoders.write().await;
            for (_, worker) in enc_map.drain() {
                worker.shutdown.store(true, Ordering::Relaxed);
                let _ = worker.handle.await;
            }
        }
        // 停止捕获线程
        if let Some(handle) = self.capture_handle.take() {
            self.capture_shutdown.store(true, Ordering::Relaxed);
            let _ = handle.await;
        }
        self.frame_broadcast = None;
        Ok(())
    }
}
