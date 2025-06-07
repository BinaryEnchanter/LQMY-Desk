// =====================================================
// main.rs
// LQMY-Desk 客户端桌面端主程序入口
//
// 功能职责：
// - 初始化 Tauri 应用
// - 提供对 Vue 前端的 RPC 接口 (tauri::command)
// - 管理 WebRTC 信令、PeerConnection 连接
// - 管理用户信息、用户状态
// - 管理全局捕获（视频流捕获、关闭流程）
// - 优雅关闭支持 (窗口关闭时清理资源)
//
// 依赖模块：
// - client：WebSocket 客户端逻辑
// - client_utils：用户管理、断连管理
// - config：全局配置与全局状态
// - video_capturer：视频采集模块
// - webrtc：WebRTC 相关处理
// =====================================================

// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod client;
mod client_utils;
mod config;
//mod error;
//mod audio_capture;
mod input_executor;
mod video_capturer;
mod webrtc;

use std::{
    env,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use client::CLOSE_NOTIFY;
use client_utils::{
    current_user::CurUsersInfo,
    disconnect::disconnect_cur_user_by_uuid,
    user_manager::{delete_user, transfer_userinfo_to_vue, update_user_category, UserInfoString},
};
use config::{reset_all_info, CONFIG, CURRENT_USERS_INFO, GLOBAL_STREAM_MANAGER, UUID};
use std::fs::File;
use webrtc::webrtc_connect::close_peerconnection;

use crate::config::APPDATA_PATH;

/// 应用状态管理结构体（用于跨命令共享运行状态）
pub struct AppState {
    pub is_running: Arc<AtomicBool>,
    pub exit_flag: Arc<AtomicBool>,
}

/// 启动客户端 WebSocket 连接 & 信令逻辑
#[tauri::command]
async fn start_server(state: tauri::State<'_, AppState>) -> Result<(), String> {
    let is_running = state.is_running.clone();
    let exit_flag = state.exit_flag.clone();

    std::thread::spawn(move || {
        let sys = actix_rt::System::new();
        is_running.store(true, Ordering::Relaxed);
        exit_flag.store(false, Ordering::Relaxed);

        println!("[CLIENT] exit_flag: {:?}", exit_flag);

        let result = sys.block_on(async { client::start_client(exit_flag.clone()).await });

        if let Err(e) = result {
            eprintln!("[CLIENT] start_client 出错: {}", e);
        }

        is_running.store(false, Ordering::Relaxed);

        // 必须在 block_on 内部调用异步函数
        sys.block_on(async {
            reset_all_info().await;
        });
    });

    Ok(())
}

/// 停止客户端连接，关闭 WebSocket 信令，清理 PeerConnection
#[tauri::command]
async fn stop_server(state: tauri::State<'_, AppState>) -> Result<(), String> {
    // 先执行异步关闭
    shutdown_caputure().await;

    // 更新状态标志
    state.is_running.store(false, Ordering::Relaxed);
    state.exit_flag.store(true, Ordering::Relaxed);

    println!(
        "Exit flag set, client should shut down soon. {:?}",
        state.exit_flag
    );

    // 通知 client 主循环退出
    CLOSE_NOTIFY.notify_one();

    // 重置全局信息
    reset_all_info().await;

    println!("[SERVER_INFO: Server stopped.]");

    Ok(())
}

/// 获取服务器配置信息，供前端显示
#[tauri::command]
async fn get_server_info(
    state: tauri::State<'_, AppState>,
) -> Result<(String, String, String, bool, CurUsersInfo), String> {
    let config = CONFIG.lock().await;
    let uuid = UUID.read().await.clone();
    let cur_users_info = CURRENT_USERS_INFO.read().await.clone();
    let is_running = state.is_running.load(Ordering::Relaxed).clone();
    Ok((
        config.server_address.clone(),
        config.connection_password.clone(),
        uuid.clone(),
        is_running,
        cur_users_info,
    ))
}

/// 获取当前所有用户信息（转换为前端可用格式）
#[tauri::command]
async fn get_user_info() -> Vec<UserInfoString> {
    let vec = transfer_userinfo_to_vue().await;
    println!("[USER LIST]传到VUE的用户信息为{:?}", vec);
    vec
}

/// 更新指定用户的用户类型
#[tauri::command]
async fn update_user_type(serial: String, usertype: String) {
    update_user_category(serial, usertype).await;
}

/// 删除指定用户信息
#[tauri::command]
async fn delete_userinfo(serial: String) {
    delete_user(serial).await
}

/// 更新服务器地址配置
#[tauri::command]
async fn update_server_addr(ipaddr: String) {
    config::update_server_addr(ipaddr).await
}

/// 断开指定 uuid 的用户连接
#[tauri::command]
async fn disconnect_by_uuid(uuid: String) {
    close_peerconnection(&uuid).await;
    disconnect_cur_user_by_uuid(&uuid).await;
}

/// 后端窗口关闭回调处理（主动调用全局关闭）
#[tauri::command]
async fn backend_close_handler() {
    shutdown_caputure().await
}

/// 撤销控制权，通知对应用户
#[tauri::command]
async fn revoke_control() {
    CURRENT_USERS_INFO.write().await.revoke_control().await;
}

/// 全局关闭视频捕获流程，断开所有 PeerConnection
#[tauri::command]
async fn shutdown_caputure() {
    let cur_users = CURRENT_USERS_INFO.read().await.usersinfo.clone();
    for curinfo in cur_users.iter() {
        close_peerconnection(&curinfo.uuid).await;
        disconnect_cur_user_by_uuid(&curinfo.uuid).await;
    }
    drop(cur_users);
    // GLOBAL_STREAM_MANAGER.write().await.shutdown().await; // 若需要可启用
    CURRENT_USERS_INFO.write().await.reset();
    println!("[SERVER]关闭捕获，全部用户断开")
}

/// 关闭流程是否已被调用（防止重复调用）
static SHUTDOWN_CALLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

// 这个编译宏可能会在 rust-analyzer的激进检错中显示语法错误，但是不影响 cargo 编译，所以可以忽略
/// 主程序入口
#[tokio::main(flavor = "multi_thread", worker_threads = 16)]
async fn main() {
    // 初始化日志输出文件
    let path = APPDATA_PATH.lock().unwrap().join("output.txt");
    let file = File::create(path).unwrap();

    #[cfg(all(windows, not(debug_assertions)))]
    {
        use std::os::windows::io::AsRawHandle;
        use std::process::Command;
    }

    // 添加 ffmpeg/bin 到 PATH，方便 ffmpeg 调用
    let mut exe_path = env::current_exe().expect("Failed to get current exe path");
    exe_path.pop(); // 移除exe文件名，保留目录
    exe_path.push("ffmpeg/bin");

    let path_env = env::var("PATH").unwrap_or_default();
    let new_path = format!("{};{}", exe_path.display(), path_env);
    env::set_var("PATH", new_path);

    // 初始化 Tauri 应用
    tauri::Builder::default()
        .on_window_event(|_window, event| {
            match event {
                tauri::WindowEvent::CloseRequested { api, .. } => {
                    // 防止多次调用 shutdown
                    if SHUTDOWN_CALLED.load(std::sync::atomic::Ordering::Relaxed) {
                        return;
                    }

                    api.prevent_close();

                    SHUTDOWN_CALLED.store(true, std::sync::atomic::Ordering::Relaxed);

                    tauri::async_runtime::spawn(async move {
                        shutdown_caputure().await;
                        std::process::exit(0);
                    });
                }
                _ => {}
            }
        })
        .manage(AppState {
            is_running: Arc::new(AtomicBool::new(false)),
            exit_flag: Arc::new(AtomicBool::new(false)),
        })
        .invoke_handler(tauri::generate_handler![
            start_server,
            stop_server,
            get_server_info,
            get_user_info,
            update_user_type,
            delete_userinfo,
            update_server_addr,
            disconnect_by_uuid,
            revoke_control,
            backend_close_handler,
            shutdown_caputure,
        ])
        .run(tauri::generate_context!())
        .expect("Failed to run Tauri application");
}
