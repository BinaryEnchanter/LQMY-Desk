use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};

use crate::config::get_userinfo_path;
use crate::log_println;

use super::dialog::show_confirmation_dialog;
use std::collections::HashMap;
use std::fs;
use std::sync::Mutex;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum UserType {
    Blacklist,
    Normal,
    Trusted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInfo {
    pub device_name: String,
    pub device_id: String,
    //hashed_password: String, //unnecessary
    pub user_type: UserType,
}

// 全局存储所有用户信息，启动服务时会从json文件读取，运行时实时更新改变量和本地消息

lazy_static! {
    pub static ref USER_LIST: Mutex<HashMap<String, UserInfo>> = Mutex::new(load_devices());
}

/// 读取本地存储的设备信息
fn load_devices() -> HashMap<String, UserInfo> {
    if let Ok(data) = fs::read_to_string(get_userinfo_path()) {
        log_println!("[USER_LIST:成功从{:?}读取用户信息]", get_userinfo_path());
        serde_json::from_str(&data).unwrap_or_else(|_| HashMap::new())
    } else {
        log_println!("[USER_LIST:路径下没有json文件,用户信息表初始化为空]");
        HashMap::new()
    }
}

/// 保存设备信息到本地
fn save_devices() {
    let devices = USER_LIST.lock().unwrap();
    if let Ok(json) = serde_json::to_string(&*devices) {
        let _ = fs::write(get_userinfo_path(), json);
    }
}

/// 根据序列号搜索
pub async fn get_user_by_serial(serial_number: &str) -> Option<UserInfo> {
    let users = USER_LIST.lock().unwrap();
    users.get(serial_number).cloned()
}

/// 添加新设备
pub async fn add_device(device_name: &str, device_id: &str) {
    {
        let mut devices = USER_LIST.lock().unwrap();
        if devices.contains_key(device_id) {
            log_println!("[USER_LIST:该设备信息已存在，无法再次添加]");
            return;
        }
        devices.insert(
            device_id.to_string(),
            UserInfo {
                device_name: device_name.to_string(),
                device_id: device_id.to_string(),
                user_type: UserType::Normal, // 默认普通用户
            },
        );
    }
    save_devices();
    log_println!("[USER_LIST:已添加设备{:?}到普通用户]", device_name);
}

#[derive(Debug, Serialize)]
pub struct UserInfoString {
    pub device_name: String,
    pub device_id: String,
    pub user_type: String,
}
pub async fn transfer_userinfo_to_vue() -> Vec<UserInfoString> {
    load_devices();
    let userlist = USER_LIST.lock().unwrap();
    userlist
        .values()
        .map(|info| UserInfoString {
            device_id: info.device_id.clone(),
            device_name: info.device_name.clone(),
            user_type: match info.user_type {
                UserType::Trusted => "trusted".to_string(),
                UserType::Normal => "regular".to_string(),
                UserType::Blacklist => "blacklist".to_string(),
            },
        })
        .collect()
}

pub async fn update_user_category(serial: String, usertype: String) {
    let mut users = USER_LIST.lock().unwrap();
    {
        let user = users.get_mut(&serial).unwrap();

        let user_type = match usertype.as_str() {
            "trusted" => "可信",
            "regular" => "普通",
            "blacklist" => "黑名单",
            _ => "未知",
        };
        let msg = format!(
            "是否将用户 {:?} 类别修改为 {:?}",
            user.device_name, user_type
        );
        if !show_confirmation_dialog("更改用户类别", &msg) {
            return;
        }
    }

    let _res = if let Some(user) = users.get_mut(&serial) {
        user.user_type = match usertype.as_str() {
            "trusted" => UserType::Trusted,
            "regular" => UserType::Normal,
            "blacklist" => UserType::Blacklist,
            _ => {
                log_println!("[USER INFO]未定义的用户类型{:?}", &usertype);
                return;
            }
        };

        log_println!(
            "[USER LIST]成功更新用户{:?}类型为'{:?}'",
            user.device_id,
            user.user_type
        );
        Ok(())
    } else {
        log_println!("[USER LIST]更新用户类型失败");
        Err(())
    };
    drop(users);
    save_devices();
}

pub async fn delete_user(serial: String) {
    let mut users = USER_LIST.lock().unwrap();
    let removed = users.remove_entry(&serial);
    drop(users);
    if let Some(rem) = removed {
        save_devices();
        log_println!("[USER_INFO]用户信息{:?}删除", rem)
    } else {
        log_println!("[USER_INFO]设备{:?}不存在，删除失败", &serial)
    }
}
