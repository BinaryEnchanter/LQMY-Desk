use crate::config::APPDATA_PATH;
use std::fs::OpenOptions;
use std::io::Write;

pub fn log_to_file(message: &str) {
    if let Ok(path) = APPDATA_PATH.lock() {
        let file_path = path.join("output.txt");
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(file_path) {
            // 使用系统时间避免 chrono 依赖
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            if let Err(_) = writeln!(file, "[{}] {}", now, message) {
                // 写入失败时的处理
            }
            let _ = file.flush();

            // debug 模式下同时输出到控制台
            #[cfg(debug_assertions)]
            println!("{}", message);
        }
    }
}

#[macro_export]
macro_rules! log_println {
    ($($arg:tt)*) => {
        crate::macros::log_to_file(&format!($($arg)*));
    };
}
