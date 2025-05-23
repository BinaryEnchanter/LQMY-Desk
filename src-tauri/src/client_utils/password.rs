use crate::config::CONFIG;
use rand::{thread_rng, Rng};

/**
 * 生成连接口令的内部函数，口令长度由 .take()的参数决定
 * 不是pub，只能由fn generate_connection_password()调用 */
#[inline]
async fn generate_password() -> String {
    let mut rng = thread_rng();
    let number: u32 = rng.gen_range(10_000_000..100_000_000);
    let password = format!("{:?}", number);
    password
}
/**
 * 设置连接口令
 */
pub async fn generate_connection_password() {
    let password = generate_password().await;
    let mut config = CONFIG.lock().unwrap();
    config.connection_password = password.clone();
    println!("Generated connection password: {:?}", password); // 打印或将口令发送给电脑端
}

/**
 * 验证手机端口令
 */
pub async fn verify_password(input_password: &str) -> bool {
    let config = CONFIG.lock().unwrap();
    input_password == config.connection_password
}
