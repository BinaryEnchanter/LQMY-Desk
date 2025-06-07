//! =======================================================
//! 文件名: cur_users.rs
//! 模块功能:
//!     - 管理当前连接的用户信息 (CurUsersInfo)
//!     - 支持控制用户切换、控制权撤销、用户增删、查找等功能
//!     - 提供控制消息分类与权限检查缓存
//! 作者: 李昶毅
//! 日期: 2025-05-20
//!         5.23 添加对控制用户的管理
//!         5.24 死锁，取消对全局变量的引用
//!  
//!     
//! =======================================================

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    client::{PENDING, SEND_NOTIFY},
    client_utils::{dialog::show_confirmation_dialog, user_manager::get_user_by_serial},
    config::{CURRENT_USERS_INFO, UUID},
};

use super::user_manager::UserType;

/// 当前连接用户列表信息
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CurUsersInfo {
    /// 最大连接数
    max: usize,

    /// 当前控制权指针：
    /// - 0..=max-1: 表示当前持有控制权的用户下标
    /// - >= max: 表示当前无控制用户
    pub pointer: usize,

    /// 当前用户信息列表
    pub usersinfo: Vec<CurInfo>,
}

impl CurUsersInfo {
    /// 创建新的 CurUsersInfo
    pub fn new(max: usize) -> Self {
        Self {
            max,
            pointer: max + 1, // 初始无控制用户
            usersinfo: Vec::<CurInfo>::new(),
        }
    }

    /// 通过设备序列号升级对应用户为控制用户（需要确认）
    ///
    /// 本方法不会检测是否已有控制用户，由调用者保证逻辑一致性
    pub async fn set_ptr_by_serial(&mut self, serial: &str) -> bool {
        if let Some(this_user) = get_user_by_serial(serial).await {
            match this_user.user_type {
                UserType::Blacklist => return false,
                UserType::Normal => {
                    let seriall = serial.to_string();
                    let approved: bool = tokio::task::spawn_blocking(move || {
                        show_confirmation_dialog(
                            "控制请求",
                            &format!("是否允许来自{:?}的控制请求", seriall),
                        )
                    })
                    .await
                    .expect("blocking task panicked");

                    if !approved {
                        return false;
                    };
                }
                UserType::Trusted => {}
            }
        };

        // 更新 pointer
        let mut pointer = 0;
        for cur_info in self.usersinfo.iter() {
            if *serial != cur_info.device_id {
                pointer += 1;
            } else {
                break;
            }
        }
        self.pointer = pointer;
        pointer < self.usersinfo.len()
    }

    /// 添加新的当前用户
    ///
    /// - 若未达连接上限，则添加并打印成功日志
    /// - 否则打印失败日志
    pub fn add_new_cur_user(&mut self, new_user: &CurInfo) {
        if self.usersinfo.len() < self.max {
            self.usersinfo.push(new_user.clone());
            println!("[CONFIG] 成功添加新的用户信息：{:?}", new_user)
        } else {
            println!("[CONFIG] 失败添加新的用户信息：{:?}", new_user)
        }
    }

    /// 重置当前用户信息
    ///
    /// - pointer 置为 max，表示无控制用户
    /// - 清空用户列表
    pub fn reset(&mut self) {
        self.pointer = self.max;
        self.usersinfo = Vec::new();
    }

    /// 根据序列号查找用户是否存在
    pub fn lookup_by_serial(&self, serial: &str) -> bool {
        self.usersinfo
            .iter()
            .any(|cur_info| serial == cur_info.device_id)
    }

    /// 当前是否有空余连接位
    pub fn is_avail(&self) -> bool {
        self.usersinfo.len() < self.max
    }

    /// 删除用户（根据 uuid）
    ///
    /// - 若删除成功返回 true
    /// - 否则返回 false
    pub fn delete_by_uuid(&mut self, uuid: &str) -> bool {
        let mut target = 0;
        for cur_info in self.usersinfo.iter() {
            if cur_info.uuid != uuid {
                target += 1;
            } else {
                break;
            }
        }

        if target < self.usersinfo.len() {
            let removed = self.usersinfo.swap_remove(target);
            println!("[CURUSER] 连接用户信息删除：{:?}", removed);
            true
        } else {
            println!("[CURUSER] 连接用户信息删除失败：{:?}", uuid);
            false
        }
    }

    /// 是否已有控制用户
    pub fn has_controller(&self) -> bool {
        self.pointer < self.usersinfo.len()
    }

    /// 判断指定 uuid 是否是当前控制用户
    pub fn is_controller_by_uuid(&self, uuid: String) -> bool {
        if self.pointer < self.usersinfo.len() {
            self.usersinfo[self.pointer].uuid == uuid
        } else {
            false
        }
    }

    /// 撤销当前控制用户
    pub async fn revoke_control(&mut self) {
        if self.pointer >= self.max {
            return;
        }

        let result = CrtlAns {
            status: "100".to_string(),
            body: "控制权取回".to_string(),
        };

        let uuid = UUID.read().await.clone();
        let reply = json!({
            "type": "message",
            "target_uuid": self.usersinfo[self.pointer].uuid,
            "from": uuid,
            "payload": json!(result),
        });

        drop(uuid);

        let mut pending = PENDING.lock().unwrap();
        pending.push(reply.clone());
        drop(pending);

        SEND_NOTIFY.notify_one();

        // 更新 pointer
        self.pointer = self.max
    }
}

/// 单个当前用户信息
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CurInfo {
    /// 设备名
    pub device_name: String,

    /// 设备序列号
    pub device_id: String,

    /// 用户类别
    pub user_type: UserType,

    /// WebSocket UUID
    pub uuid: String,
}

/// 控制请求 (客户端发来请求控制)
#[derive(Debug, Deserialize)]
pub struct CrtlReq {
    pub jwt: String,
    pub uuid: String,
    pub device_serial: String,
}

/// 控制请求回复
#[derive(Debug, Serialize)]
pub struct CrtlAns {
    pub status: String,
    pub body: String,
}

/// 控制消息分类
#[derive(Debug, Clone)]
pub enum ControlMessage {
    MouseMove(Value),
    MouseClick(Value),
    KeyPress(Value),
    Other(Value),
}

impl ControlMessage {
    /// 从 JSON 消息中解析为 ControlMessage 类型
    pub fn from_json(json: Value) -> Self {
        if let Some(msg_type) = json.get("type").and_then(|t| t.as_str()) {
            match msg_type {
                "mouse_move" => Self::MouseMove(json),
                "mouse_click" => Self::MouseClick(json),
                "key_press" => Self::KeyPress(json),
                _ => Self::Other(json),
            }
        } else {
            Self::Other(json)
        }
    }

    /// 判断该消息是否可丢弃
    ///
    /// - 鼠标移动消息可以丢弃
    /// - 其他消息需要保证送达
    pub fn is_droppable(&self) -> bool {
        matches!(self, Self::MouseMove(_))
    }
}

/// 控制权限检查缓存
pub struct ControllerCache {
    /// 当前用户 uuid
    pub uuid: String,

    /// 是否为控制用户
    pub is_controller: bool,

    /// 上次检查时间
    pub last_check: Instant,
}

impl ControllerCache {
    /// 创建新的缓存对象
    pub fn new(uuid: String) -> Self {
        Self {
            uuid,
            is_controller: false,
            last_check: Instant::now() - Duration::from_secs(1), // 强制首次检查
        }
    }

    /// 当前缓存是否有效
    pub fn is_valid(&self) -> bool {
        self.last_check.elapsed() < Duration::from_millis(200) // 200ms 缓存
    }

    /// 检查并更新缓存
    pub async fn check_and_update(&mut self) -> bool {
        if !self.is_valid() {
            self.is_controller = CURRENT_USERS_INFO
                .read()
                .await
                .is_controller_by_uuid(self.uuid.clone());
            self.last_check = Instant::now();
        }
        self.is_controller
    }
}
