use bytes::{Bytes, BytesMut};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::UdpSocket;
use tokio::task;
use webrtc::media::Sample;
use webrtc::peer_connection::RTCPeerConnection;
use webrtc::rtp::packet::Packet;
use webrtc::rtp_transceiver::rtp_codec::RTCRtpCodecCapability;
use webrtc::track::track_local::track_local_static_rtp::TrackLocalStaticRTP;
use webrtc::track::track_local::track_local_static_sample::TrackLocalStaticSample;
use webrtc::track::track_local::TrackLocalWriter;
use webrtc::util::Unmarshal;
use webrtc::Error;

/// 给定一个已存在的 PeerConnection 和 UDP 端口，向它添加一个 H264Track
/// 然后从 UDP socket 读取 RTP 包写入该 Track。
pub async fn start_webrtc_video_stream_on_pc(
    pc: Arc<RTCPeerConnection>,
    udp_port: u16,
    mode: String,
) -> Arc<TrackLocalStaticSample> {
    // 1. 根据模式选择 codec 和 fmtp
    let (mime, fmt) = match mode.as_str() {
        "low_latency" => ("video/H264", "max-fr=30;max-fs=360"),
        "high_quality" => (
            "video/H264",
            "profile-level-id=42e01f;level-asymmetry-allowed=1",
        ),
        _ => ("video/H264", "max-fr=24;max-fs=480"), // balanced
    };

    // 2. 用 StaticRTP 而非 StaticSample
    // let video_track = Arc::new(TrackLocalStaticRTP::new(
    //     RTCRtpCodecCapability {
    //         mime_type: "video/H264".into(),
    //         sdp_fmtp_line: "profile-level-id=42e01f;packetization-mode=1;level-asymmetry-allowed=1;x-google-max-bitrate=1500".into(),
    //         clock_rate: 90000,
    //         ..Default::default()
    //     },
    //     "video".into(),
    //     "rust-video".into(),
    // ));
    let video_track = Arc::new(TrackLocalStaticSample::new(
        RTCRtpCodecCapability {
            mime_type: "video/H264".into(),
            sdp_fmtp_line: "profile-level-id=42e01f;level-asymmetry-allowed=1;packetization-mode=1"
                .into(),
            clock_rate: 90000,
            ..Default::default()
        },
        "video".into(),      // track ID
        "rust-video".into(), // stream ID
    ));
    // 3. 添加到 PeerConnection（会在下一次协商时被包含到 SDP 里）
    pc.add_track(video_track.clone()).await;
    video_track
    // // 4. 绑定 UDP Socket
    // let bind_addr: SocketAddr = format!("127.0.0.1:{}", udp_port).parse().unwrap();
    // let socket = UdpSocket::bind(bind_addr).await.unwrap();

    // // 5. 启动后台任务，不断读包并写入 RTP Track
    // task::spawn(async move {
    //     let mut buf = vec![0u8; 1500];
    //     loop {
    //         if let Ok((size, _peer)) = socket.recv_from(&mut buf).await {
    //             // let mut packet_buf = BytesMut::from(&buf[..size]);
    //             // if let Ok(pkt) = Packet::unmarshal(&mut packet_buf) {
    //             //     // StaticRTP 提供了 write_rtp
    //             //     //log_println!("[RTP] Write RTP packet: {} bytes", pkt.marshal_size());
    //             //     //let res = video_track.write_rtp(&pkt).await;
    //             //     //log_println!("[RTP]videotrack write {:?}", res);
    //             //     //log_println!("[RTP] Sending RTP packet size{:?}", size);
    //             //     // let mut out_buf = vec![0u8; 1500];

    //             //     let sample = Sample {
    //             //         data: pkt.payload.clone(), // 注意不是整个 RTP packet，只是 payload
    //             //         duration: Duration::from_millis(33),
    //             //         ..Default::default()
    //             //     };
    //             //      let _res = video_track.write_sample(&sample).await;
    //             //     //log_println!("[VIDEOTRACK]写入RTC:{:?}", res)
    //             // } else {
    //             //     log_println!("[VIDEOSTREAM]packet_buf{:?}", packet_buf)
    //             // }

    //             let nalu = &buf[..size];
    //             let sample = Sample {
    //                 data: Bytes::copy_from_slice(nalu),
    //                 duration: Duration::from_millis(33), // 30fps
    //                 ..Default::default()
    //             };
    //             if let Err(err) = video_track.write_sample(&sample).await {
    //                 log_println!("[H264Sample] write_sample err: {}", err);
    //             }
    //         } else {
    //             log_println!("[VIDEOSTREAM]packet_buf{:?}", buf)
    //         }
    //     }
    // });

    // log_println!("[VIDEOTRACK] 启动 UDP→WebRTC 推流 @{}", udp_port);
    //Ok(())
}
