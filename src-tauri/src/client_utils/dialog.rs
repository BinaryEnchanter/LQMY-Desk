//! =======================================================
//! 文件名: diaglog.rs
//! 模块功能:
//!     - 弹出 确认/取消 对话框
//!     - 弹出 确认 对话框
//! 作者: 李昶毅
//! 日期: 2025-04-20
//!         使用需要放到 阻塞线程，否则会使程序崩溃
//!  
//!     
//! =======================================================

use rfd::{MessageDialog, MessageDialogResult};

pub fn show_confirmation_dialog(title: &str, message: &str) -> bool {
    let result = MessageDialog::new()
        .set_title(title)
        .set_description(message)
        .set_buttons(rfd::MessageButtons::OkCancel) // 显示 “确认/取消” 按钮
        .show(); // 阻塞，等待用户点击
    result == MessageDialogResult::Ok
}

pub async fn show_iknow_dialog(title: &str, message: &str) {
    MessageDialog::new()
        .set_title(title)
        .set_description(message)
        .set_buttons(rfd::MessageButtons::Ok) // 显示 “确” 按钮
        .show(); // 阻塞，等待用户点击
}
