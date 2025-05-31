// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
mod client;
mod client_utils;
mod config;
mod error;
mod video_capturer;
mod webrtc;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use client::CLOSE_NOTIFY;
use client_utils::{
    current_user::CurUsersInfo,
    disconnect::disconnect_cur_user_by_uuid,
    user_manager::{delete_user, transfer_userinfo_to_vue, update_user_category, UserInfoString},
};
use config::{reset_all_info, CONFIG, CURRENT_USERS_INFO, PEER_CONNECTION, UUID};
use video_capturer::ffmpeg::end_screen_capture;
use webrtc::webrtc_connect::close_peerconnection;

//use actix_web::{web, App, HttpServer, HttpResponse};
//use tauri::Manager;

pub struct AppState {
    pub is_running: Arc<AtomicBool>,
    pub exit_flag: Arc<AtomicBool>,
}

#[tauri::command]
fn start_server(state: tauri::State<AppState>) {
    let is_running = state.is_running.clone();
    let exit_flag = state.exit_flag.clone();
    std::thread::spawn(move || {
        let sys = actix_rt::System::new();
        is_running.store(true, Ordering::Relaxed);
        exit_flag.store(false, std::sync::atomic::Ordering::Relaxed);
        println!("[CLIENT]exit_flag:{:?}", exit_flag);
        let _ = sys.block_on(async { client::start_client(exit_flag).await });

        is_running.store(false, Ordering::Relaxed);
        reset_all_info();
        //reset_cur_user();
    });
}

#[tauri::command]
fn stop_server(state: tauri::State<AppState>) {
    state.is_running.store(false, Ordering::Relaxed);
    // 重置连接状况，将连接者信息清楚
    state
        .exit_flag
        .store(true, std::sync::atomic::Ordering::Relaxed);
    println!(
        "Exit flag set, client should shut down soon.{:?}",
        state.exit_flag
    );
    CLOSE_NOTIFY.notify_one();

    //reset_cur_user();

    reset_all_info();
    println!("[SERVER_INFO: Server stopped.");
}

#[tauri::command]
fn get_server_info(state: tauri::State<AppState>) -> (String, String, String, bool, CurUsersInfo) {
    let config = CONFIG.lock().unwrap();
    let uuid = UUID.lock().unwrap();
    println!(
        "[SERVER_INFO: Acquiring addr {:?} & password {:?} & uuid {:?}]",
        config.server_address, config.connection_password, uuid
    );
    //let cur_user = CURRENT_USER.lock().unwrap();
    let cur_users_info = CURRENT_USERS_INFO.lock().unwrap().clone();
    let is_running = state.is_running.clone();
    (
        config.server_address.clone(),
        config.connection_password.clone(),
        // cur_user.device_name.clone(),
        // cur_user.device_id.clone(),
        // format!("{:?}", cur_user.user_type),
        uuid.clone(),
        is_running.load(Ordering::Relaxed),
        cur_users_info,
    )
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
    config::update_server_addr(ipaddr)
}

#[tauri::command]
async fn disconnect_by_uuid(uuid: String) {
    close_peerconnection(&uuid).await;
    disconnect_cur_user_by_uuid(&uuid);
}
#[tauri::command]
async fn backend_close_handler() {
    // 1. 先收集所有连接的克隆
    let connections_to_close = {
        let pcs = PEER_CONNECTION.lock().unwrap();
        pcs.iter()
            .map(|(uuid, pc)| (uuid.clone(), pc.clone()))
            .collect::<Vec<_>>()
    }; // MutexGuard 在这里被释放

    // 2. 逐个关闭连接
    for (uuid, pc) in connections_to_close {
        if let Err(e) = pc.close().await {
            println!("[CLOSE]uuid:{:?}关闭失败，{:?}", uuid, e);
        }
    }

    // 3. 清空 HashMap
    {
        let mut pcs = PEER_CONNECTION.lock().unwrap();
        pcs.clear();
    }

    // 关闭录屏
    end_screen_capture(true);
}
#[tauri::command]
/// 本地函数,不会向对方发消息
async fn revoke_control() {
    CURRENT_USERS_INFO.lock().unwrap().revoke_control();
}
#[tokio::main(flavor = "multi_thread", worker_threads = 16)]
async fn main() {
    tauri::Builder::default()
        // .on_window_event(|window, event| {
        //     match event {
        //         tauri::WindowEvent::CloseRequested { .. } => {
        //             //backend_close_handler().await;
        //             window.close();
        //         }
        //         _ => {} //todo
        //     }
        // })
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
        ])
        .run(tauri::generate_context!())
        .expect("Failed to run Tauri application");
}
