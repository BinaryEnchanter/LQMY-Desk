// /*
// 重构示例：
// - 用一个中央捕获任务将 H264 样本推入 tokio broadcast 通道
// - 各个工作线程（或任务）通过订阅广播通道，接收同样的样本并调用 write_sample

// Cargo.toml:

// [dependencies]
// ffmpeg-next = "4.3"
// tokio = { version = "1", features = ["full"] }
// webrtc = "0.6"
// bytes = "1"

// */
// use std::sync::Arc;
// use tokio::sync::{broadcast, Mutex};
// use webrtc::media::Sample;
// use webrtc::track::track_local::track_local_static_sample::TrackLocalStaticSample;
// use ffmpeg_next as ffmpeg;
// use ffmpeg::format::context::Input;
// use ffmpeg::codec;
// use ffmpeg::software::scaling::{context::Context as Scaler, flag::Flags};
// use ffmpeg::util::frame::video::Video;

// /// 启动中央捕获任务，将编码后的 H264 样本广播给所有订阅者
// pub async fn start_capture_broadcaster(
//     buffer_size: usize,
// ) -> ffmpeg::Result<broadcast::Sender<Sample>> {
//     ffmpeg::init()?;

//     // 打开 gdigrab 设备
//     let mut ictx = Input::with_format("gdigrab", "desktop")?.input()?;
//     ictx.set_options(&[("framerate","30"),("draw_mouse","1")])?;
//     ictx.open(None)?;

//     let input_stream = ictx.streams().best(ffmpeg::media::Type::Video).unwrap();
//     let stream_index = input_stream.index();
//     let mut decoder = codec::context::Context::from_parameters(input_stream.parameters())?
//         .decoder().video()?;

//     // 初始化 H264 编码器
//     let global_header = ictx.format().flags().contains(ffmpeg::format::flag::Flags::GLOBAL_HEADER);
//     let mut encoder = codec::encoder::video::Video::new(
//         codec::Id::H264,
//         decoder.width(), decoder.height(),
//     )?;
//     encoder.set_time_base((1,30)); encoder.set_frame_rate(Some((30,1)));
//     if global_header { encoder.set_flags(codec::flag::Flags::GLOBAL_HEADER); }
//     let mut encoder = encoder.open_as(codec::Id::H264)?;

//     // 像素格式转换
//     let mut scaler = Scaler::get(
//         decoder.format(), decoder.width(), decoder.height(),
//         encoder.format(), encoder.width(), encoder.height(),
//         Flags::BILINEAR,
//     )?;

//     // 创建广播通道
//     let (tx, _) = broadcast::channel(buffer_size);

//     // 捕获与编码循环
//     tokio::spawn(async move {
//         let mut packet = ffmpeg::Packet::empty();
//         let mut frame_dec = Video::empty();
//         let mut frame_enc = Video::empty();

//         loop {
//             if let Ok(_) = ictx.read(&mut packet) {
//                 if packet.stream_index() != stream_index { continue; }
//                 if decoder.send_packet(&packet).is_err() { break; }

//                 while decoder.receive_frame(&mut frame_dec).is_ok() {
//                     scaler.run(&frame_dec, &mut frame_enc).unwrap();
//                     encoder.send_frame(&frame_enc).unwrap();

//                     let mut enc_pkt = ffmpeg::Packet::empty();
//                     while encoder.receive_packet(&mut enc_pkt).is_ok() {
//                         let sample = Sample {
//                             data: bytes::Bytes::from(enc_pkt.data().to_vec()),
//                             duration: std::time::Duration::from_millis(33),
//                             ..Default::default()
//                         };
//                         // 广播给所有订阅者
//                         let _ = tx.send(sample);
//                     }
//                 }
//             } else {
//                 break;
//             }
//         }
//     });

//     Ok(tx)
// }

// /// 订阅广播并写入 TrackLocalStaticSample
// pub async fn spawn_writer_from_broadcaster(
//     mut rx: broadcast::Receiver<Sample>,
//     track: Arc<TrackLocalStaticSample>,
// ) {
//     tokio::spawn(async move {
//         while let Ok(sample) = rx.recv().await {
//             if let Err(err) = track.write_sample(&sample).await {
//                 log_println!("write_sample error: {}", err);
//                 break;
//             }
//         }
//     });
// }

// // 使用示例：
// // #[tokio::main]
// // async fn main() {
// //     let broadcaster = start_capture_broadcaster(16).await.unwrap();
// //     // 比如有多个 track
// //     let track1 = Arc::new(...);
// //     let track2 = Arc::new(...);
// //     let rx1 = broadcaster.subscribe();
// //     let rx2 = broadcaster.subscribe();
// //     spawn_writer_from_broadcaster(rx1, track1).await;
// //     spawn_writer_from_broadcaster(rx2, track2).await;
// // }
