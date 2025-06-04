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

//use actix_web::{web, App, HttpServer, HttpResponse};
//use tauri::Manager;

pub struct AppState {
    pub is_running: Arc<AtomicBool>,
    pub exit_flag: Arc<AtomicBool>,
}

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

    CLOSE_NOTIFY.notify_one();

    // 重置全局信息
    reset_all_info().await;

    println!("[SERVER_INFO: Server stopped.]");

    Ok(())
}
#[tauri::command]
async fn get_server_info(
    state: tauri::State<'_, AppState>,
) -> Result<(String, String, String, bool, CurUsersInfo), String> {
    let config = CONFIG.lock().await;
    let uuid = UUID.read().await.clone();
    println!(
        "[SERVER_INFO: Acquiring addr {:?} & password {:?} & uuid {:?}]",
        config.server_address, config.connection_password, uuid
    );
    //let cur_user = CURRENT_USER.lock().unwrap();
    let cur_users_info = CURRENT_USERS_INFO.read().await.clone();
    let is_running = state.is_running.load(Ordering::Relaxed).clone();
    Ok((
        config.server_address.clone(),
        config.connection_password.clone(),
        // cur_user.device_name.clone(),
        // cur_user.device_id.clone(),
        // format!("{:?}", cur_user.user_type),
        uuid.clone(),
        is_running,
        cur_users_info,
    ))
}

#[tauri::command]
async fn get_user_info() -> Vec<UserInfoString> {
    let vec = transfer_userinfo_to_vue().await;
    println!("[USER LIST]传到VUE的用户信息为{:?}", vec);
    vec
}
#[tauri::command]
async fn update_user_type(serial: String, usertype: String) {
    update_user_category(serial, usertype).await;
}
#[tauri::command]
async fn delete_userinfo(serial: String) {
    delete_user(serial).await
}

#[tauri::command]
async fn update_server_addr(ipaddr: String) {
    config::update_server_addr(ipaddr).await
}

#[tauri::command]
async fn disconnect_by_uuid(uuid: String) {
    close_peerconnection(&uuid).await;
    disconnect_cur_user_by_uuid(&uuid).await;
}
#[tauri::command]
async fn backend_close_handler() {
    shutdown_caputure().await
}
#[tauri::command]
/// 撤销控制，会向对方发消息
async fn revoke_control() {
    CURRENT_USERS_INFO.write().await.revoke_control().await;
}
#[tauri::command]
async fn shutdown_caputure() {
    let cur_users = CURRENT_USERS_INFO.read().await.usersinfo.clone();
    for curinfo in cur_users.iter() {
        close_peerconnection(&curinfo.uuid).await;
        disconnect_cur_user_by_uuid(&curinfo.uuid).await;
    }
    drop(cur_users);
    //GLOBAL_STREAM_MANAGER.write().await.shutdown().await;
    CURRENT_USERS_INFO.write().await.reset();
    println!("[SERVER]关闭捕获，全部用户断开")
}

static SHUTDOWN_CALLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[tokio::main(flavor = "multi_thread", worker_threads = 16)]
async fn main() {
    let path = APPDATA_PATH.lock().unwrap().join("output.txt");
    let file = File::create(path).unwrap();

    #[cfg(all(windows, not(debug_assertions)))]
    {
        use std::os::windows::io::AsRawHandle;
        use std::process::Command;
    }
    let mut exe_path = env::current_exe().expect("Failed to get current exe path");
    exe_path.pop(); // 移除exe文件名，保留目录

    // 添加到 PATH 环境变量
    let path_env = env::var("PATH").unwrap_or_default();
    let new_path = format!("{};{}", exe_path.display(), path_env);
    env::set_var("PATH", new_path);
    tauri::Builder::default()
        .on_window_event(|_window, event| {
            match event {
                tauri::WindowEvent::CloseRequested { api, .. } => {
                    // 检查是否已经调用过shutdown
                    if SHUTDOWN_CALLED.load(std::sync::atomic::Ordering::Relaxed) {
                        return; // 如果已经调用过，直接返回让窗口正常关闭
                    }

                    // 阻止默认关闭
                    api.prevent_close();

                    // 设置标志，防止重复调用
                    SHUTDOWN_CALLED.store(true, std::sync::atomic::Ordering::Relaxed);

                    // 异步执行cleanup
                    //let window_clone = window.clone();
                    tauri::async_runtime::spawn(async move {
                        shutdown_caputure().await;
                        // cleanup完成后退出程序
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
