use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    client::{PENDING, SEND_NOTIFY},
    client_utils::disconnect,
    config::{CURRENT_USERS_INFO, UUID},
};

use super::auth::validate_jwt;

#[derive(Debug, Deserialize)]
/// 被动断连
pub struct DisconnectReq {
    pub jwt: String,
    pub device_serial: String,
}

impl DisconnectReq {
    pub fn verify(&self) -> bool {
        validate_jwt(&self.jwt)
    }
}

#[derive(Debug, Serialize)]
pub struct Disconnect {
    pub cmd: String,
}

/// 这个函数不仅要删除当前连接用户的信息，还要返回一个消息告诉对方关闭连接了
pub fn disconnect_cur_user_by_uuid(uuid: &str) {
    //删除连接信息
    if CURRENT_USERS_INFO.lock().unwrap().delete_by_uuid(uuid) {
        // 告知对方
        let res = Disconnect {
            cmd: "disconnect".to_owned(),
        };

        let reply = json!({
            "type": "message",
            "target_uuid": uuid,
            "from":UUID.lock().unwrap().clone(),
            "payload": json!(res),
        });
        PENDING.lock().unwrap().push(reply.clone());
        SEND_NOTIFY.notify_one();
    }
}
