use lazy_static::lazy_static;
use rand::{distr::Alphanumeric, Rng};

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::{env, path::PathBuf};
use tokio::sync::RwLock;

use webrtc::ice_transport::ice_candidate::RTCIceCandidateInit;
use webrtc::peer_connection::RTCPeerConnection;

use crate::audio_capture::winapi::WebRTCAudioSystem;
use crate::client_utils::current_user::CurUsersInfo;
use crate::client_utils::user_manager::{UserInfo, UserType};
use crate::video_capturer::assembly::MultiStreamManager;
pub const NO_CONNECTION_INDENTIFIER: &str = "!@#$%^&*()";
// 存储全局信息的结构体
pub struct Config {
    pub server_address: String,      // 电脑开放的端口
    pub connection_password: String, // 生成的连接口令
}

lazy_static! {
    // 服务器信息 websocket IP/ 连接口令
    pub static ref CONFIG: Mutex<Config> = Mutex::new(Config {
        server_address: env::var("SERVER_ADDRESS").unwrap_or_else(|_| "wss://localhost:9876".to_string()),
        connection_password: "Uninitia".to_string(),
    });
    // 当前连接用户信息
    pub static ref CURRENT_USER:Mutex<UserInfo>=Mutex::new(UserInfo{
        device_name:"".to_string(),
        device_id:NO_CONNECTION_INDENTIFIER.to_string(),
        user_type:UserType::Normal
    });
    // 当前连接用户信息向量
    pub static ref CURRENT_USERS_INFO:Mutex<CurUsersInfo>=Mutex::new(CurUsersInfo::new(5));

    pub static ref APPDATA_PATH:Mutex<PathBuf>=Mutex::new(load_storage_path());
    // JWT加密密钥，每次启动不一样
    pub static ref JWT_KEY:Mutex<String>=Mutex::new(generate_jwt_key());
    // 标识jwt是本次启动生成的
    pub static ref THIS_TIME:Mutex<String>=Mutex::new(generate_jwt_key());
    //已经移到user_manage.rs管理
    //pub static ref DEVICE_LIST: Mutex<HashMap<String, DeviceInfo>> = Mutex::new(HashMap::new());// 没有放到CONFIG，为了减少不必要的并发访问冲突
    // 中转站分配的uuid
    pub static ref UUID:Mutex<String>=Mutex::new("尚未连接服务器".to_string());
    // 全局 PeerConnection 存储：session_id -> PeerConnection
    pub static ref PEER_CONNECTION: Mutex<HashMap<String, Arc<RTCPeerConnection>>> = Mutex::new(HashMap::new());
    // 全局候选列表存储：session_id -> Vec<RTCIceCandidateInit>
    pub static ref CANDIDATES: Mutex<HashMap<String, Vec<RTCIceCandidateInit>>> = Mutex::new(HashMap::new());
    // websocket 客户端的连接，全局共享
    //pub static ref WS_SENDER:Arc<Mutex<Option<awc::BoxedSocket>>>=Arc::new(Mutex::new(None));

    // 视频流管理器
    pub static ref GLOBAL_STREAM_MANAGER: Arc<RwLock<MultiStreamManager>> =
        Arc::new(RwLock::new(MultiStreamManager::new()));

    // 音频流管理器
    pub static ref GLOBAL_AUDIO_MANAGER: Arc<RwLock<WebRTCAudioSystem>> =
        Arc::new(RwLock::new(WebRTCAudioSystem::new()));

}
fn load_storage_path() -> PathBuf {
    "E:/WHU/SoftwareEngineering/GroupWork/LQMY-Desk".into()
}

pub fn get_userinfo_path() -> PathBuf {
    let path = APPDATA_PATH.lock().unwrap();
    path.join("user_data.json")
}

fn generate_jwt_key() -> String {
    let password: String = rand::rng()
        .sample_iter(&Alphanumeric)
        .take(8) // 这里修改口令长度
        .map(char::from)
        .collect();
    password
}

// pub fn update_cur_user(info: &web::Json<AuthRequest>, usertype: UserType) {
//     let mut cur_user = CURRENT_USER.lock().unwrap();

//     cur_user.device_id = info.device_serial.clone();
//     cur_user.device_name = info.device_name.clone();
//     cur_user.user_type = usertype;
//     println!(
//         "[SERVER_INFO]连接用户信息更新：设备名：{:?}，设备序列号：{:?}，用户类型：{:?}",
//         cur_user.device_name, cur_user.device_id, cur_user.user_type
//     );
// }

// pub fn reset_cur_user() {
//     let mut cur_user = CURRENT_USER.lock().unwrap();

//     cur_user.device_id = NO_CONNECTION_INDENTIFIER.to_string();
//     cur_user.device_name = "".to_string();
//     cur_user.user_type = UserType::Normal;
//     println!(
//         "[SERVER_INFO]连接用户信息重置为：设备名：{:?}，设备序列号：{:?}，用户类型：{:?}",
//         cur_user.device_name, cur_user.device_id, cur_user.user_type
//     );
// }

pub fn update_uuid(uuid: &str) {
    let mut cur_uuid = UUID.lock().unwrap();
    *cur_uuid = uuid.to_string();
    println!("[CLIENT]服务器分配的uuid：{:?}", *cur_uuid)
}

pub fn update_server_addr(ipaddr: String) {
    let mut config = CONFIG.lock().unwrap();
    config.server_address = ipaddr;
    println!("[CLIENT]所连服务器信息改变为:{:?}", config.server_address);
}

pub fn reset_all_info() {
    let mut config = CONFIG.lock().unwrap();
    config.connection_password = "Uninitia".to_string();
    let mut uuid = UUID.lock().unwrap();
    *uuid = "尚未连接服务器".to_string();
    CURRENT_USERS_INFO.lock().unwrap().reset();
    println!("[CONFIG]口令、用户与UUID重置")
}

// pub fn add_to_cur_user_vec(new_user: &UserInfo) {
//     let mut cur_users_info = CURRENT_USERS_INFO.lock().unwrap();
//     cur_users_info.usersinfo.push(new_user.clone());
//     println!("[CONFIG]添加新连接用户信息：{:?}", new_user)
// }
