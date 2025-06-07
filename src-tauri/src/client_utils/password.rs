//! =======================================================
//! 文件名: password.rs
//! 模块功能:
//!     - 生成口令
//! 作者: 李昶毅
//! 日期: 2025-04-8
//!         使用需要放到 阻塞线程，否则会使程序崩溃
//! 5.23 优化体验，口令改为数字串
//!     
//! =======================================================
use crate::config::CONFIG;
use rand::{rng, Rng};

/**
 * 生成连接口令的内部函数，口令长度由 .take()的参数决定
 * 不是pub，只能由fn generate_connection_password()调用 */
#[inline]
async fn generate_password() -> String {
    let mut rng = rng();
    let number: u32 = rng.random_range(10_000_000..100_000_000);
    let password = format!("{:?}", number);
    password
}
/**
 * 设置连接口令
 */
pub async fn generate_connection_password() {
    let password = generate_password().await;
    let mut config = CONFIG.lock().await;
    config.connection_password = password.clone();
    println!("Generated connection password: {:?}", password); // 打印或将口令发送给电脑端
}

// /**
//  * 验证手机端口令
//  */
// pub async fn verify_password(input_password: &str) -> bool {
//     let config = CONFIG.lock().await;
//     input_password == config.connection_password
// }
