use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    client::{PENDING, SEND_NOTIFY},
    client_utils::{dialog::show_confirmation_dialog, user_manager::get_user_by_serial},
    config::UUID,
};

use super::user_manager::UserType;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CurUsersInfo {
    max: usize,         // 默认最大连接数
    pub pointer: usize, // 0..=max-1 表示有控制权的用户， 此外无意义

    pub usersinfo: Vec<CurInfo>,
}
impl CurUsersInfo {
    pub fn new(max: usize) -> Self {
        Self {
            max,
            pointer: max + 1,
            usersinfo: Vec::<CurInfo>::new(),
        }
    }

    /// 通过CurInfo来升级对应的用户为控制用户，但是这里没有检测是否已有控制用户
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

    // /// 设置ptr为max，表示已经到连接上限了
    // pub fn set_ptr_as_max(&mut self){
    //     self.pointer=self.max;
    // }

    /// 添加新的连接用户信息
    pub fn add_new_cur_user(&mut self, new_user: &CurInfo) {
        if self.usersinfo.len() < self.max {
            self.usersinfo.push(new_user.clone());
            println!("[CONFIG]成功添加新的用户信息：{:?}", new_user)
        } else {
            println!("[CONFIG]失败添加新的用户信息：{:?}", new_user)
        }
    }

    /// 重置信息
    pub fn reset(&mut self) {
        self.pointer = self.max;
        self.usersinfo = Vec::new();
    }

    /// 序列号确认信息存在与否
    pub fn lookup_by_serial(&self, serial: &str) -> bool {
        let mut res = false;
        for cur_info in self.usersinfo.iter() {
            res |= serial == cur_info.device_id;
        }
        res
    }

    /// 是否有空余
    pub fn is_avail(&self) -> bool {
        self.usersinfo.len() < self.max
    }

    /// 删除用户
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
            println!("[CURUSER]连接用户信息删除：{:?}", removed);
            true
        } else {
            println!("[CURUSER]连接用户信息删除失败：{:?}", uuid);
            false
        }
    }

    /// 检查是否已经有控制对象
    pub fn has_controller(&self) -> bool {
        self.pointer < self.usersinfo.len()
    }

    /// 判断是不是控制对象
    pub fn is_controller_by_uuid(&self, uuid: String) -> bool {
        let pointer = self.pointer;
        if pointer < self.usersinfo.len() {
            self.usersinfo[pointer].uuid == uuid
        } else {
            false
        }
    }

    pub async fn revoke_control(&mut self) {
        let result = CrtlAns {
            status: "100".to_string(),
            body: "控制权取回".to_string(),
        };
        let uuid = UUID.read().await.clone();
        let reply = json!({
            "type": "message",
            "target_uuid": self.usersinfo[self.pointer].uuid,
            "from":uuid,
            "payload": json!(result),
        });
        drop(uuid);
        let mut pending = PENDING.lock().unwrap();
        pending.push(reply.clone());
        drop(pending);
        SEND_NOTIFY.notify_one();
        self.pointer = self.max
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CurInfo {
    pub device_name: String,
    pub device_id: String,
    pub user_type: UserType,
    pub uuid: String,
}

#[derive(Debug, Deserialize)]
pub struct CrtlReq {
    pub jwt: String,
    pub uuid: String,
    pub device_serial: String,
}
#[derive(Debug, Serialize)]
pub struct CrtlAns {
    pub status: String,
    pub body: String,
}
// pub async fn handle_control_request(controlreq: CrtlReq) -> CrtlAns {
//     let curusers = CURRENT_USERS_INFO.lock().unwrap();
//     if !curusers.has_controller() {
//         curusers.set_ptr_by_serial(&controlreq.device_serial);
//     }
// }
