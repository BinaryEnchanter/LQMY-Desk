use crate::client::{PENDING, SEND_NOTIFY};
use crate::client_utils::current_user::{ControlMessage, ControllerCache};
use crate::config::{CURRENT_USERS_INFO, GLOBAL_STREAM_MANAGER, ICE_BUFFER, PEER_CONNECTION, UUID};
use crate::input_executor::input::decode_and_dispatch;
use crate::video_capturer::ffmpeg::*;

use enigo::{Enigo, Settings};
use futures_util::FutureExt;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::mpsc::{self, unbounded_channel};
use webrtc::ice_transport::ice_credential_type::RTCIceCredentialType;

use std::sync::Arc;
use webrtc::track::track_local::track_local_static_sample::TrackLocalStaticSample;

use webrtc::data_channel::RTCDataChannel;

use webrtc::rtp_transceiver::RTCPFeedback;

use webrtc::api::media_engine::MediaEngine;
use webrtc::api::APIBuilder;

use webrtc::ice_transport::ice_candidate::RTCIceCandidateInit;

use webrtc::ice_transport::ice_server::RTCIceServer;

use webrtc::peer_connection::configuration::RTCConfiguration;
use webrtc::peer_connection::peer_connection_state::RTCPeerConnectionState;
use webrtc::peer_connection::sdp::session_description::RTCSessionDescription;
use webrtc::rtp_transceiver::rtp_codec::RTCRtpCodecCapability;

#[derive(Debug, Deserialize)]
pub struct JWTOfferRequest {
    pub client_uuid: String,
    pub sdp: String,
    pub mode: String, // "low_latency", "balanced", "high_quality"
    pub jwt: String,
}

#[derive(Serialize)]
pub struct AnswerResponse {
    pub client_uuid: String,
    pub sdp: String,
}

#[derive(Deserialize)]
pub struct JWTCandidateRequest {
    pub client_uuid: String,
    pub candidate: String,
    pub sdp_mid: Option<String>,
    pub sdp_mline_index: Option<u16>,
    pub jwt: String,
}

#[derive(Serialize)]
pub struct CandidateResponse {
    pub candidates: RTCIceCandidateInit,
}

// 初始 Offer/Answer，返回 AnswerResponse
pub async fn handle_webrtc_offer(offer: &JWTOfferRequest) -> AnswerResponse {
    println!("[WEBRTC]准备启动");
    let client_uuid = &offer.client_uuid;

    // 1. 初始化 MediaEngine 并注册 codecs
    let mut m = MediaEngine::default();
    if let Err(e) = m.register_default_codecs() {
        let msg = format!("MediaEngine 注册失败: {:?}", e);
        return AnswerResponse {
            client_uuid: client_uuid.clone(),
            sdp: msg,
        };
    }
    let api = APIBuilder::new().with_media_engine(m).build();

    // 2. 创建 PeerConnection
    let config = RTCConfiguration {
        ice_servers: vec![
            RTCIceServer {
                urls: vec![
                    "stun:stun.l.google.com:19302".into(),
                    "stun:stun.qq.com:3478".into(),
                ],
                ..Default::default()
            },
            // Metered.ca TURN 服务器配置
            RTCIceServer {
                urls: vec![
                    "turn:a.relay.metered.ca:80".to_owned(),
                    "turn:a.relay.metered.ca:80?transport=tcp".to_owned(),
                    "turn:a.relay.metered.ca:443".to_owned(),
                    "turn:a.relay.metered.ca:443?transport=tcp".to_owned(),
                ],
                username: "a0999ebcb5073b9b1f7d80fc".to_owned(), // 替换为你的用户名
                credential: "U0bKVWmLzwkHJ0fx".to_owned(),       // 替换为你的密码
                credential_type: RTCIceCredentialType::Password,
            },
        ],
        ..Default::default()
    };
    //let pc = api.new_peer_connection(config).await?;
    let pc = match api.new_peer_connection(config).await {
        Ok(pc) => Arc::new(pc),
        Err(e) => {
            let msg = format!("PeerConnection 创建失败: {:?}", e);
            return AnswerResponse {
                client_uuid: client_uuid.clone(),
                sdp: msg,
            };
        }
    };
    PEER_CONNECTION
        .write()
        .await
        .insert(client_uuid.clone(), pc.clone());
    println!("[PCS]当前连接{:?}", client_uuid.clone());
    // 3. (可选) negotiationneeded 调试
    pc.on_negotiation_needed(Box::new(|| {
        println!("[WEBRTC] negotiationneeded");
        Box::pin(async {})
    }));

    let video_track = Arc::new(TrackLocalStaticSample::new(
        RTCRtpCodecCapability {
            mime_type: "video/H264".into(),
            sdp_fmtp_line: "level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42E02A"
                .into(),
            clock_rate: 90000,
            rtcp_feedback: vec![
                RTCPFeedback {
                    typ: "nack".to_owned(),
                    parameter: "".to_owned(),
                },
                RTCPFeedback {
                    typ: "nack".to_owned(),
                    parameter: "pli".to_owned(), // picture loss indication
                },
                RTCPFeedback {
                    typ: "goog-remb".to_owned(), // optional, for bandwidth estimation
                    parameter: "".to_owned(),
                },
                RTCPFeedback {
                    typ: "ccm".to_owned(),
                    parameter: "fir".to_owned(),
                },
                RTCPFeedback {
                    typ: "transport-cc".to_owned(), // 传输拥塞控制
                    parameter: "".to_owned(),
                },
            ],
            ..Default::default()
        },
        "video".into(),      // track ID
        "rust-video".into(), // stream ID
    ));

    pc.add_track(video_track.clone()).await.unwrap();
    let clien_uuidd = client_uuid.clone();
    let (tx, mut rx) = tokio::sync::mpsc::channel::<ControlMessage>(1000);
    tokio::spawn(async move {
        let mut enigo = Enigo::new(&Settings::default()).unwrap();
        let mut message_buffer = Vec::with_capacity(50);
        let mut last_process = Instant::now();

        while let Some(msg) = rx.recv().await {
            message_buffer.push(msg);

            // 批量处理逻辑：缓冲区满或超时
            let should_process =
                message_buffer.len() >= 20 || last_process.elapsed() > Duration::from_millis(16); // ~60fps

            if should_process {
                // 处理消息，对鼠标移动做去重
                let mut processed_messages = Vec::new();
                let mut last_mouse_move: Option<ControlMessage> = None;

                for msg in message_buffer.drain(..) {
                    match &msg {
                        ControlMessage::MouseMove(_) => {
                            // 只保留最后一个鼠标移动消息
                            last_mouse_move = Some(msg);
                        }
                        _ => {
                            // 先处理积累的鼠标移动
                            if let Some(mouse_msg) = last_mouse_move.take() {
                                processed_messages.push(mouse_msg);
                            }
                            processed_messages.push(msg);
                        }
                    }
                }

                // 处理剩余的鼠标移动
                if let Some(mouse_msg) = last_mouse_move {
                    processed_messages.push(mouse_msg);
                }

                // 执行控制操作
                for msg in processed_messages {
                    match msg {
                        ControlMessage::MouseMove(json) => decode_and_dispatch(&mut enigo, &json),
                        ControlMessage::MouseClick(json) => decode_and_dispatch(&mut enigo, &json),
                        ControlMessage::KeyPress(json) => decode_and_dispatch(&mut enigo, &json),
                        ControlMessage::Other(json) => decode_and_dispatch(&mut enigo, &json),
                    }
                }

                last_process = Instant::now();
            }
        }
    });

    // 4. 优化的DataChannel处理
    pc.on_data_channel(Box::new({
        let uuid = clien_uuidd.clone();
        let tx = tx.clone();
        move |dc: Arc<RTCDataChannel>| {
            println!("[WEBRTC] 收到远端 DataChannel: label = {}", dc.label());

            dc.on_message(Box::new({
                let uuid = uuid.clone();
                let tx = tx.clone();
                move |msg| {
                    let data = msg.data.clone();
                    let uuid = uuid.clone();
                    let tx = tx.clone();

                    // 立即返回，避免阻塞WebRTC事件循环
                    tokio::spawn(async move {
                        // 使用本地缓存减少全局锁竞争
                        static mut CACHE: Option<ControllerCache> = None;
                        let cache = unsafe {
                            if CACHE.is_none() {
                                CACHE = Some(ControllerCache::new(uuid.clone()));
                            }
                            CACHE.as_mut().unwrap()
                        };

                        // 权限检查
                        if !cache.check_and_update().await {
                            return;
                        }

                        // 快速解析和分发
                        if let Ok(text) = std::str::from_utf8(&data) {
                            if let Ok(json_val) = serde_json::from_str::<Value>(text) {
                                let control_msg = ControlMessage::from_json(json_val);

                                // 使用try_send避免阻塞
                                match tx.try_send(control_msg.clone()) {
                                    Ok(_) => {}
                                    Err(mpsc::error::TrySendError::Full(_)) => {
                                        // 通道满了，如果是可丢弃消息就丢弃
                                        if control_msg.is_droppable() {
                                            println!("[WARN] 丢弃鼠标移动消息，通道已满");
                                        } else {
                                            // 重要消息，等待发送
                                            if let Err(e) = tx.send(control_msg).await {
                                                eprintln!("[ERROR] 重要消息发送失败: {:?}", e);
                                            }
                                        }
                                    }
                                    Err(mpsc::error::TrySendError::Closed(_)) => {
                                        eprintln!("[ERROR] 控制消息通道已关闭");
                                        return;
                                    }
                                }
                            }
                        }
                    });

                    Box::pin(async {})
                }
            }));

            Box::pin(async {})
        }
    }));

    // 7. 收集本地 ICE 候选
    {
        let uuid = client_uuid.clone();
        pc.on_ice_candidate(Box::new(move |opt| {
            let uuid = uuid.clone(); // capture外部变量

            async move {
                if let Some(c) = opt {
                    if let Ok(json) = c.to_json() {
                        let init = RTCIceCandidateInit {
                            candidate: json.candidate,
                            sdp_mid: json.sdp_mid,
                            sdp_mline_index: json.sdp_mline_index,
                            username_fragment: None,
                        };

                        let my_uuid = UUID.read().await.clone();
                        let res = send_ice_candidate(init);
                        let payload = json!({"cmd":"candidate","value":res});
                        let reply = json!({
                            "type": "message",
                            "target_uuid": uuid,
                            "from": my_uuid,
                            "payload": payload,
                        });

                        {
                            let mut pending = PENDING.lock().unwrap();
                            pending.push(reply.clone());
                        }

                        SEND_NOTIFY.notify_one();
                        println!("[CLIENT] RTC返回ICE：{:?}", reply);
                    }
                }
            }
            .boxed()
        }));
    }

    // 8. ICE 连接成功后推流
    {
        //let pc2 = pc.clone();
        pc.on_ice_connection_state_change(Box::new(move |state| {
            println!("[WEBRTC]连接状态改变，ICEState：{:?}", state);
            //monitor_video_send_stats(pc2.clone());
            Box::pin(async {})
        }));
    }

    {
        let pc2 = pc.clone();
        let client_uuid2 = client_uuid.clone();
        let mode2 = offer.mode.clone();
        pc.on_peer_connection_state_change(Box::new(move |state| {
            println!("[WEBRTC]连接状态改变，ConnectionState： {:?}", state);

            if state == RTCPeerConnectionState::Connected {
                println!("✅ DTLS 握手成功");
                let video_track2 = video_track.clone();
                let client_uuid3 = client_uuid2.clone();
                let mode3 = mode2.clone();
                tokio::task::spawn(async move {
                    // 5. 启动后台任务，不断读包并写入 RTP Track
                    if let Err(e) = GLOBAL_STREAM_MANAGER.write().await.start_capture().await {
                        println!("[WEBRTC]启动视频失败，{:?}", e)
                    };
                    let q = select_mode(&mode3, &client_uuid3);

                    if let Err(e) = GLOBAL_STREAM_MANAGER
                        .read()
                        .await
                        .add_webrtc_track(&client_uuid3, video_track2, q)
                        .await
                    {
                        println!("[WEBRTC]写入track失败，{:?}", e)
                    };
                });
            } else if state == RTCPeerConnectionState::Closed {
                let pc3 = pc2.clone();
                let client_uuid3 = client_uuid2.clone();
                tokio::task::spawn(async move {
                    if let Err(e) = pc3.close().await {
                        println!("[RTC]关闭peerconnection失败{:?}", e)
                    } else {
                        println!("[RTC]被动关闭{:?}的连接", client_uuid3)
                    }
                    if let Err(e) = GLOBAL_STREAM_MANAGER
                        .write()
                        .await
                        .remove_track(&client_uuid3)
                        .await
                    {
                        println!("[FFMPEG]关闭失败 {:?}", e)
                    };
                    // GLOBAL_STREAM_MANAGER.write().await.check_shutdown().await;
                    //end_screen_capture(false);
                });
            } else if state == RTCPeerConnectionState::Disconnected {
                let pc3 = pc2.clone();
                let client_uuid3 = client_uuid2.clone();
                tokio::task::spawn(async move {
                    tokio::time::sleep(Duration::from_secs(3)).await;
                    if pc3.connection_state() != RTCPeerConnectionState::Disconnected {
                        return;
                    }

                    if let Err(e) = pc3.close().await {
                        println!("[RTC]关闭peerconnection失败{:?}", e)
                    } else {
                        println!("[RTC]被动关闭{:?}的连接", client_uuid3)
                    };
                    if let Err(e) = GLOBAL_STREAM_MANAGER
                        .write()
                        .await
                        .remove_track(&client_uuid3)
                        .await
                    {
                        println!("[FFMPEG]关闭失败 {:?}", e)
                    };
                    // GLOBAL_STREAM_MANAGER.write().await.check_shutdown().await;
                    //end_screen_capture(false);
                });
            }
            Box::pin(async {})
        }));
    }
    // 9. SDP Offer/Answer
    let remote = RTCSessionDescription::offer(offer.sdp.clone()).unwrap();
    pc.set_remote_description(remote).await.unwrap();
    let answer = pc.create_answer(None).await.unwrap();
    if let Err(e) = pc.set_local_description(answer.clone()).await {
        eprint!("[LOCAL DES]{:?}", e)
    };

    // 10. 保存并返回

    AnswerResponse {
        client_uuid: client_uuid.clone(),
        sdp: answer.sdp,
    }
}

// 客户端上传远端 ICE 候选，直接返回结果字符串
pub async fn handle_ice_candidate(req: JWTCandidateRequest) -> String {
    let init = RTCIceCandidateInit {
        candidate: req.candidate.clone(),
        sdp_mid: req.sdp_mid.clone(),
        sdp_mline_index: req.sdp_mline_index,
        username_fragment: None,
    };

    if let Some(pc) = PEER_CONNECTION.read().await.get(&req.client_uuid) {
        // PC存在，直接添加ICE候选
        pc.add_ice_candidate(init).await.unwrap();
        "ICE 注入成功".into()
    } else {
        // PC不存在，缓冲ICE候选
        ICE_BUFFER
            .write()
            .await
            .entry(req.client_uuid.clone())
            .or_insert_with(Vec::new)
            .push(init);
        "ICE 已缓冲".into()
    }
}

// 创建PeerConnection后调用此函数处理缓冲的ICE
pub async fn flush_buffered_ice(client_uuid: &str) {
    if let Some(pc) = PEER_CONNECTION.read().await.get(client_uuid) {
        if let Some(candidates) = ICE_BUFFER.write().await.remove(client_uuid) {
            for candidate in candidates {
                let _ = pc.add_ice_candidate(candidate).await;
            }
        }
    };
}
// 客户端拉取本地 ICE 候选，直接返回 CandidateResponse
// pub fn send_ice_candidate(uuid: &str) -> CandidateResponse {
//     //let client_uuid = info.get("client_uuid").cloned().unwrap_or_default();
//     // let res = CANDIDATES
//     //     .lock()
//     //     .unwrap()
//     //     .get(uuid)
//     //     .cloned()
//     //     .unwrap_or_default();
//     let mut lock = CANDIDATES.lock().unwrap();
//     let cands = lock.remove(uuid).unwrap_or_default();
//     CandidateResponse { candidates: cands }
// }
#[inline]
pub fn send_ice_candidate(candi: RTCIceCandidateInit) -> CandidateResponse {
    CandidateResponse { candidates: candi }
}

use tokio::time::{interval, Duration, Instant};
use webrtc::peer_connection::RTCPeerConnection;

pub fn monitor_video_send_stats(pc: Arc<RTCPeerConnection>) {
    tokio::spawn(async move {
        let mut ticker = interval(Duration::from_secs(30));
        loop {
            ticker.tick().await;

            // 直接获取统计报告
            let report = pc.get_stats().await;
            println!("{:?}", report.reports);
            for (_id, stat) in report.reports {
                if let Ok(json) = serde_json::to_value(&stat) {
                    // 筛选出视频的发送统计
                    if json.get("type") == Some(&Value::String("outbound-rtp".into()))
                        && json.get("mediaType") == Some(&Value::String("video".into()))
                    {
                        let bytes_sent =
                            json.get("bytesSent").and_then(|v| v.as_u64()).unwrap_or(0);
                        let packets_sent = json
                            .get("packetsSent")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        let frames_encoded = json
                            .get("framesEncoded")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        println!(
                            "[STATS] bytes_sent = {}, packets_sent = {}, frames_encoded = {}",
                            bytes_sent, packets_sent, frames_encoded
                        );
                    }
                }
            }
        }
    });
}

/// 关闭指定peerconnection，以uuid为索引
pub async fn close_peerconnection(client_uuid: &str) {
    // First, check if the connection exists and get a clone of the Arc
    let pc_option = {
        let pcs = PEER_CONNECTION.write().await;
        if pcs.is_empty() {
            println!("[CLOSE PC]指定用户的RTC连接不存在{:?}", client_uuid);
            return;
        }
        pcs.get(client_uuid).cloned() // Clone the Arc, not the connection itself
    }; // MutexGuard is dropped here

    // Now work with the cloned Arc outside the mutex
    if let Some(pc) = pc_option {
        if let Err(e) = GLOBAL_STREAM_MANAGER
            .write()
            .await
            .remove_track(client_uuid)
            .await
        {
            println!("[FFMPEG]删除失败{:?}", e)
        };
        if let Err(e) = pc.close().await {
            println!("[CLOSE PC]指定用户的RTC关闭失败，{:?},{:?}", e, client_uuid);
            return;
        }

        // Remove from the HashMap after successful close
        {
            let mut pcs = PEER_CONNECTION.write().await;
            pcs.remove(client_uuid);
        } // MutexGuard is dropped here

        println!("[CLOSE PC]指定用户的RTC关闭成功，{:?}", client_uuid);

        //end_screen_capture(false);
    } else {
        println!("[CLOSE PC]指定用户的RTC连接不存在{:?}", client_uuid);
    }
}
fn select_mode(mode: &str, _client_uuid: &str) -> EncodingParams {
    let res = match mode {
        "high" => EncodingParams {
            width: 1920,
            height: 1080,
            fps: 45,
            bitrate: 10000000,
        },

        "low" => EncodingParams {
            width: 1920,
            height: 1080,
            fps: 30,
            bitrate: 5000000,
        },
        _ => EncodingParams {
            width: 1280,
            height: 720,
            fps: 30,
            bitrate: 2000000,
        },
    };
    //println!("{:?}", res);
    res
}
