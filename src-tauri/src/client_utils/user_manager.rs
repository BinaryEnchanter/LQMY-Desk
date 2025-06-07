//! =======================================================
//! 文件名: user_manager.rs
//! 模块功能:
//!     - 管理本地存储的用户信息表 (USER_LIST)
//!     - 提供用户信息的增删改查接口
//!     - 提供用户类别转换
//!     - 启动时从本地 json 文件自动加载，实时更新本地 json
//! 作者: xxx
//! 日期: 2025-04.20
//!
//! =======================================================

use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};

use crate::config::get_userinfo_path;

use super::dialog::show_confirmation_dialog;
use std::collections::HashMap;
use std::fs;
use std::sync::Mutex;

/// 用户类别
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum UserType {
    /// 黑名单用户，禁止连接
    Blacklist,

    /// 普通用户，需要申请控制权限
    Normal,

    /// 可信用户，自动获得控制权限
    Trusted,
}

/// 单个用户信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInfo {
    /// 设备名
    pub device_name: String,

    /// 设备序列号
    pub device_id: String,

    /// 用户类别
    pub user_type: UserType,
}

// 全局存储所有用户信息
//
// 启动服务时会从 json 文件读取
// 运行时实时更新该变量和本地 json 文件
lazy_static! {
    pub static ref USER_LIST: Mutex<HashMap<String, UserInfo>> = Mutex::new(load_devices());
}

/// 读取本地存储的设备信息
///
/// - 若文件存在，尝试反序列化 json
/// - 若文件不存在，初始化为空表
fn load_devices() -> HashMap<String, UserInfo> {
    if let Ok(data) = fs::read_to_string(get_userinfo_path()) {
        println!("[USER_LIST: 成功从 {:?} 读取用户信息]", get_userinfo_path());
        serde_json::from_str(&data).unwrap_or_else(|_| HashMap::new())
    } else {
        println!("[USER_LIST: 路径下没有 json 文件，用户信息表初始化为空]");
        HashMap::new()
    }
}

/// 保存设备信息到本地 json 文件
fn save_devices() {
    let devices = USER_LIST.lock().unwrap();
    if let Ok(json) = serde_json::to_string(&*devices) {
        let _ = fs::write(get_userinfo_path(), json);
    }
}

/// 根据序列号搜索用户
///
/// 返回：
/// - Some(UserInfo) ：找到对应用户
/// - None ：未找到
pub async fn get_user_by_serial(serial_number: &str) -> Option<UserInfo> {
    let users = USER_LIST.lock().unwrap();
    users.get(serial_number).cloned()
}

/// 添加新设备
///
/// - 若设备序列号已存在，不重复添加
/// - 默认类别为 Normal
pub async fn add_device(device_name: &str, device_id: &str) {
    {
        let mut devices = USER_LIST.lock().unwrap();
        if devices.contains_key(device_id) {
            println!("[USER_LIST: 该设备信息已存在，无法再次添加]");
            return;
        }
        devices.insert(
            device_id.to_string(),
            UserInfo {
                device_name: device_name.to_string(),
                device_id: device_id.to_string(),
                user_type: UserType::Normal,
            },
        );
    }
    save_devices();
    println!("[USER_LIST: 已添加设备 {:?} 到普通用户]", device_name);
}

/// 用于传输给 Vue 的用户信息结构
#[derive(Debug, Serialize)]
pub struct UserInfoString {
    /// 设备名
    pub device_name: String,

    /// 设备序列号
    pub device_id: String,

    /// 用户类别（以字符串传输到前端）
    pub user_type: String,
}

/// 将当前用户信息表转换为 Vue 可用格式
///
/// 返回：Vec<UserInfoString>
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

/// 修改用户类别
///
/// - serial：设备序列号
/// - usertype：目标类别 ("trusted", "regular", "blacklist")
///
/// ⚠️ 需要用户确认弹窗
pub async fn update_user_category(serial: String, usertype: String) {
    let mut users = USER_LIST.lock().unwrap();
    {
        let user = users.get_mut(&serial).unwrap();

        let user_type_str = match usertype.as_str() {
            "trusted" => "可信",
            "regular" => "普通",
            "blacklist" => "黑名单",
            _ => "未知",
        };

        let msg = format!(
            "是否将用户 {:?} 类别修改为 {:?}",
            user.device_name, user_type_str
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
                println!("[USER INFO] 未定义的用户类型 {:?}", &usertype);
                return;
            }
        };

        println!(
            "[USER LIST] 成功更新用户 {:?} 类型为 '{:?}'",
            user.device_id, user.user_type
        );
        Ok(())
    } else {
        println!("[USER LIST] 更新用户类型失败");
        Err(())
    };

    drop(users);
    save_devices();
}

/// 删除用户
///
/// - serial：设备序列号
pub async fn delete_user(serial: String) {
    let mut users = USER_LIST.lock().unwrap();
    let removed = users.remove_entry(&serial);
    drop(users);

    if let Some(rem) = removed {
        save_devices();
        println!("[USER_INFO] 用户信息 {:?} 删除", rem);
    } else {
        println!("[USER_INFO] 设备 {:?} 不存在，删除失败", &serial);
    }
}
