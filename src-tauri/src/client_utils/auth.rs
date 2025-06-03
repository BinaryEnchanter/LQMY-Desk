use super::current_user::CurInfo;
use super::dialog::show_confirmation_dialog;
use super::user_manager::UserType;
use crate::client_utils::user_manager::{add_device, get_user_by_serial};
use crate::config::{CONFIG, CURRENT_USERS_INFO, JWT_KEY, THIS_TIME};

use chrono;
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Mutex;

#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    device_serial: String,
    this_time: String,
    exp: usize, // 过期时间
}

#[derive(Debug, Deserialize, Serialize)]
pub struct AuthRequest {
    pub device_name: String,
    pub device_serial: String,
    pub password: String,
    pub uuid: String,
}

#[derive(Debug, Serialize)]
pub struct AuthResponse {
    pub status: String,
    pub body: String,
}

lazy_static! {
    /// 正在等待用户确认的 device_serial 集合
    static ref CONFIRMING: Mutex<HashSet<String>> = Mutex::new(HashSet::new());
}
/// websocket连接,处理逻辑: 1.判断服务端是否空闲
///                        2.根据用户类别处理
///                             （1）黑名单：直接拒绝
///                             （2）信任：返回jwt，不验证口令，更新CURRENT——USER
///                             （3）普通：口令正确，并且ui确认，则返回jwt，更新CURRENT——USER
///                             （4）新用户：口令正确，并且ui确认，则返回jwt，更新CURRENT——USER，添加新用户信息
pub async fn authenticate(info: AuthRequest) -> AuthResponse {
    {
        let cur_users = CURRENT_USERS_INFO.read().await;
        if !cur_users.is_avail() {
            println!(
                "[SERVER_INFO]当前已连接设备上限,来自{:?}的连接请求直接拒绝",
                info.device_name
            );
            //return HttpResponse::Forbidden().body("已有设备连接，连接被拒绝");
            return AuthResponse {
                status: "403".to_owned(),
                body: "连接被拒绝".to_owned(),
            };
        }
    }
    //let users = USER_LIST.lock().unwrap();
    let this_user = get_user_by_serial(&info.device_serial).await;

    match this_user {
        // 黑名单用户直接拒绝
        Some(user) if user.user_type == UserType::Blacklist => {
            //HttpResponse::Forbidden().body("连接被拒绝")
            AuthResponse {
                status: "403".to_owned(),
                body: "连接被拒绝".to_owned(),
            }
        }
        // 信任用户直接返回jwt
        Some(user) if user.user_type == UserType::Trusted => {
            //update_cur_user(&info, UserType::Trusted);
            let userinfo = CurInfo {
                device_name: info.device_name.clone(),
                device_id: info.device_serial.clone(),
                user_type: UserType::Trusted,
                uuid: info.uuid.clone(),
            };
            CURRENT_USERS_INFO.write().await.add_new_cur_user(&userinfo);
            let token = generate_jwt(&info.device_serial);
            //HttpResponse::Ok().json(token)
            AuthResponse {
                status: "200".to_owned(),
                body: token,
            }
        }
        //普通用户，验证成功则返回jwt
        Some(user) if user.user_type == UserType::Normal => {
            let stored_pw = CONFIG.lock().await.connection_password.clone();
            if stored_pw == info.password {
                // 添加至连接队列

                {
                    let mut confirming = CONFIRMING.lock().unwrap();
                    if !confirming.insert(info.device_serial.clone()) {
                        // 已经有一个对话框在等此 serial 的确认
                        // 直接返回“pending”，不弹新框
                        return AuthResponse {
                            status: "202".into(),
                            body: "请求已在处理，请稍后".into(),
                        };
                    }
                }
                let msg = format!(
                    "是否允许来自{:?}({:?})的连接？",
                    info.device_name.clone(),
                    info.device_serial.clone()
                );
                let approved: bool =
                    tokio::task::spawn_blocking(move || show_confirmation_dialog("连接请求", &msg))
                        .await
                        .expect("blocking task panicked");
                // 从确认队列去除
                {
                    let mut confirming = CONFIRMING.lock().unwrap();
                    confirming.remove(&info.device_serial);
                }
                if approved {
                    //update_cur_user(&info, UserType::Normal);
                    let userinfo = CurInfo {
                        device_name: info.device_name.clone(),
                        device_id: info.device_serial.clone(),
                        user_type: UserType::Normal,
                        uuid: info.uuid.clone(),
                    };
                    CURRENT_USERS_INFO.write().await.add_new_cur_user(&userinfo);
                    let token = generate_jwt(&info.device_serial);
                    //HttpResponse::Ok().json(token)
                    AuthResponse {
                        status: "200".to_owned(),
                        body: token,
                    }
                } else {
                    //HttpResponse::Unauthorized().body("连接被拒绝")
                    AuthResponse {
                        status: "403".to_owned(),
                        body: "连接被拒绝".to_owned(),
                    }
                }
            } else {
                //HttpResponse::Unauthorized().body("连接口令错误")
                AuthResponse {
                    status: "403".to_owned(),
                    body: "连接口令错误".to_owned(),
                }
            }
        }
        //新用户，验证成功则记录信息并返回jwt
        _ => {
            let stored_pw = CONFIG.lock().await.connection_password.clone();
            println!("here");
            if stored_pw == info.password {
                {
                    let mut confirming = CONFIRMING.lock().unwrap();
                    if !confirming.insert(info.device_serial.clone()) {
                        // 已经有一个对话框在等此 serial 的确认
                        // 直接返回“pending”，不弹新框
                        return AuthResponse {
                            status: "202".into(),
                            body: "请求已在处理，请稍后".into(),
                        };
                    }
                }
                let msg = format!(
                    "是否允许来自{:?}({:?})的连接？",
                    info.device_name.clone(),
                    info.device_serial.clone()
                );

                let approved: bool =
                    tokio::task::spawn_blocking(move || show_confirmation_dialog("连接请求", &msg))
                        .await
                        .expect("blocking task panicked");
                // 从确认队列去除
                {
                    let mut confirming = CONFIRMING.lock().unwrap();
                    confirming.remove(&info.device_serial);
                }
                if approved {
                    //update_cur_user(&info, UserType::Normal);
                    let userinfo = CurInfo {
                        device_name: info.device_name.clone(),
                        device_id: info.device_serial.clone(),
                        user_type: UserType::Normal,
                        uuid: info.uuid.clone(),
                    };
                    CURRENT_USERS_INFO.write().await.add_new_cur_user(&userinfo);
                    let token = generate_jwt(&info.device_serial);
                    add_device(&info.device_name, &info.device_serial).await;
                    println!("[AUTH_INFO]生成jwt{:?}", token);
                    //HttpResponse::Ok().json(token)
                    AuthResponse {
                        status: "200".to_owned(),
                        body: token,
                    }
                } else {
                    //HttpResponse::Unauthorized().body("连接被拒绝")
                    AuthResponse {
                        status: "403".to_owned(),
                        body: "连接被拒绝".to_owned(),
                    }
                }
            } else {
                //HttpResponse::Unauthorized().body("连接口令错误")
                AuthResponse {
                    status: "403".to_owned(),
                    body: "连接口令错误".to_owned(),
                }
            }
        }
    }
}

fn generate_jwt(device_serial: &str) -> String {
    let claims = Claims {
        device_serial: device_serial.to_string(),
        this_time: THIS_TIME.lock().unwrap().clone(),
        exp: chrono::Utc::now().timestamp() as usize + 3600, // 1 小时有效
    };
    encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(JWT_KEY.lock().unwrap().as_ref()),
    )
    .unwrap()
}

pub fn validate_jwt(token: &str) -> bool {
    match decode::<Claims>(
        token,
        &DecodingKey::from_secret(JWT_KEY.lock().unwrap().as_ref()),
        &Validation::default(),
    ) {
        Ok(decoded) => {
            let this_time = THIS_TIME.lock().unwrap().to_string();
            decoded.claims.this_time == this_time
        }
        Err(_) => false,
    }
}
