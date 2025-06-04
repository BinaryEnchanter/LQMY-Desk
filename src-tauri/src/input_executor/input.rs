use enigo::{Axis, Button, Coordinate, Direction, Enigo, Key, Keyboard, Mouse, Settings};
use serde_json::Value;

use crate::log_println;

// 将字符串形式的“按键名”映射到 enigo::Key 枚举。
fn map_key(key_str: &str) -> Option<Key> {
    match key_str.to_lowercase().as_str() {
        // 数字键
        "0" => Some(Key::Num0),
        "1" => Some(Key::Num1),
        "2" => Some(Key::Num2),
        "3" => Some(Key::Num3),
        "4" => Some(Key::Num4),
        "5" => Some(Key::Num5),
        "6" => Some(Key::Num6),
        "7" => Some(Key::Num7),
        "8" => Some(Key::Num8),
        "9" => Some(Key::Num9),

        // 字母键
        "a" => Some(Key::A),
        "b" => Some(Key::B),
        "c" => Some(Key::C),
        "d" => Some(Key::D),
        "e" => Some(Key::E),
        "f" => Some(Key::F),
        "g" => Some(Key::G),
        "h" => Some(Key::H),
        "i" => Some(Key::I),
        "j" => Some(Key::J),
        "k" => Some(Key::K),
        "l" => Some(Key::L),
        "m" => Some(Key::M),
        "n" => Some(Key::N),
        "o" => Some(Key::O),
        "p" => Some(Key::P),
        "q" => Some(Key::Q),
        "r" => Some(Key::R),
        "s" => Some(Key::S),
        "t" => Some(Key::T),
        "u" => Some(Key::U),
        "v" => Some(Key::V),
        "w" => Some(Key::W),
        "x" => Some(Key::X),
        "y" => Some(Key::Y),
        "z" => Some(Key::Z),

        // 功能键
        "f1" => Some(Key::F1),
        "f2" => Some(Key::F2),
        "f3" => Some(Key::F3),
        "f4" => Some(Key::F4),
        "f5" => Some(Key::F5),
        "f6" => Some(Key::F6),
        "f7" => Some(Key::F7),
        "f8" => Some(Key::F8),
        "f9" => Some(Key::F9),
        "f10" => Some(Key::F10),
        "f11" => Some(Key::F11),
        "f12" => Some(Key::F12),

        // 控制键
        "esc" | "escape" => Some(Key::Escape),
        "enter" | "return" => Some(Key::Return),
        "backspace" => Some(Key::Backspace),
        "tab" => Some(Key::Tab),
        "caps" | "capslock" => Some(Key::CapsLock),
        "shift" => Some(Key::Shift),
        "ctrl" | "control" => Some(Key::Control),
        "alt" => Some(Key::Alt),
        "meta" | "win" | "command" => Some(Key::Meta),
        "space" => Some(Key::Space),

        "insert" => Some(Key::Insert),
        "delete" => Some(Key::Delete),
        "home" => Some(Key::Home),
        "end" => Some(Key::End),
        "pageup" => Some(Key::PageUp),
        "pagedown" => Some(Key::PageDown),

        "left" | "leftarrow" => Some(Key::LeftArrow),
        "right" | "rightarrow" => Some(Key::RightArrow),
        "up" | "uparrow" => Some(Key::UpArrow),
        "down" | "downarrow" => Some(Key::DownArrow),

        // 标点符号，用 Unicode 表示
        "`" => Some(Key::Unicode('`')),
        "-" => Some(Key::Unicode('-')),
        "=" => Some(Key::Unicode('=')),
        "[" => Some(Key::Unicode('[')),
        "]" => Some(Key::Unicode(']')),
        "\\" => Some(Key::Unicode('\\')),
        ";" => Some(Key::Unicode(';')),
        "'" => Some(Key::Unicode('\'')),
        "," => Some(Key::Unicode(',')),
        "." => Some(Key::Unicode('.')),
        "/" => Some(Key::Unicode('/')),

        // 常用多媒体或系统键
        "printscr" | "print" => Some(Key::PrintScr),
        "pause" => Some(Key::Pause),
        "volumeup" => Some(Key::VolumeUp),
        "volumedown" => Some(Key::VolumeDown),
        "volumemute" => Some(Key::VolumeMute),

        _ => None,
    }
}

// 解码并执行鼠标相关操作（包括“touchpad”和“mouse_xxx”类型）
//
// 拖拽示例：
// { "type":"touchpad", "event":"drag_start", "position":{"x":0.7,"y":0.4} }
// { "type":"touchpad", "event":"drag_update", "delta":{"dx":-0.003,"dy":0.002} }
// { "type":"touchpad", "event":"drag_end" }
//
// 滚动示例（不是真滚轮，而是光标移动）：
// { "type":"touchpad", "event":"scroll", "delta":{"dx":0.0,"dy":0.01} }
//
pub fn decode_mouse_event(enigo: &mut Enigo, json: &Value) {
    // 先拿出 "cmd" 或 "type"，都转为小写
    let cmd_lower = json
        .get("cmd")
        .and_then(|v| v.as_str())
        .map(|s| s.to_lowercase())
        .or_else(|| {
            json.get("type")
                .and_then(|v| v.as_str())
                .map(|s| s.to_lowercase())
        });

    let cmd = match cmd_lower {
        Some(c) => c,
        None => return,
    };

    match cmd.as_str() {
        // —— Touchpad Click ——
        "touchpad" => {
            if let Some(event_type) = json.get("event").and_then(|v| v.as_str()) {
                match event_type {
                    "click" => {
                        // 拿归一化的点击位置
                        if let Some(pos) = json.get("position") {
                            if let (Some(rel_x), Some(rel_y)) = (
                                pos.get("x").and_then(|v| v.as_f64()),
                                pos.get("y").and_then(|v| v.as_f64()),
                            ) {
                                // 先把光标移动到点击位置
                                match enigo.main_display() {
                                    Ok((w, h)) => {
                                        let abs_x = (rel_x * w as f64) as i32;
                                        let abs_y = (rel_y * h as f64) as i32;
                                        let _ = enigo.move_mouse(abs_x, abs_y, Coordinate::Abs);
                                    }
                                    Err(_) => {
                                        // fallback 分辨率 1920×1080
                                        let abs_x = (rel_x * 1920.0) as i32;
                                        let abs_y = (rel_y * 1080.0) as i32;
                                        let _ = enigo.move_mouse(abs_x, abs_y, Coordinate::Abs);
                                    }
                                }

                                // 判断是否双击 (doubleTap)，双击则用右键，否则按 "button" 字段决定
                                let is_double = json
                                    .get("doubleTap")
                                    .and_then(|v| v.as_bool())
                                    .unwrap_or(false);
                                let button_str = json
                                    .get("button")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("left");

                                let btn = if is_double {
                                    // 双击 → 右键
                                    Button::Right
                                } else {
                                    // 单击 → 根据前端发来的按钮类型
                                    match button_str {
                                        "left" => Button::Left,
                                        "right" => Button::Right,
                                        "middle" => Button::Middle,
                                        "back" => Button::Back,
                                        "forward" => Button::Forward,
                                        _ => Button::Left,
                                    }
                                };

                                // 执行点击
                                let _ = enigo.button(btn, Direction::Click);
                            }
                        }
                    }

                    "drag_start" => {
                        // 拖拽开始：先移动光标，然后按下左键
                        if let Some(pos) = json.get("position") {
                            if let (Some(rel_x), Some(rel_y)) = (
                                pos.get("x").and_then(|v| v.as_f64()),
                                pos.get("y").and_then(|v| v.as_f64()),
                            ) {
                                match enigo.main_display() {
                                    Ok((w, h)) => {
                                        let abs_x = (rel_x * w as f64) as i32;
                                        let abs_y = (rel_y * h as f64) as i32;
                                        let _ = enigo.move_mouse(abs_x, abs_y, Coordinate::Abs);
                                        let _ = enigo.button(Button::Left, Direction::Press);
                                    }
                                    Err(_) => {
                                        let abs_x = (rel_x * 1920.0) as i32;
                                        let abs_y = (rel_y * 1080.0) as i32;
                                        let _ = enigo.move_mouse(abs_x, abs_y, Coordinate::Abs);
                                        let _ = enigo.button(Button::Left, Direction::Press);
                                    }
                                }
                            }
                        }
                    }

                    "drag_update" => {
                        // 拖拽更新：移动增量 = delta * (屏幕宽度、高度)
                        if let Some(delta) = json.get("delta") {
                            if let (Some(dx), Some(dy)) = (
                                delta.get("dx").and_then(|v| v.as_f64()),
                                delta.get("dy").and_then(|v| v.as_f64()),
                            ) {
                                // 拿到当前主显示器分辨率
                                if let Ok((w, h)) = enigo.main_display() {
                                    let dx_pix = (dx * w as f64) as i32;
                                    let dy_pix = (dy * h as f64) as i32;
                                    let _ = enigo.move_mouse(dx_pix, dy_pix, Coordinate::Rel);
                                } else {
                                    // fallback 用 1920×1080
                                    let dx_pix = (dx * 1920.0) as i32;
                                    let dy_pix = (dy * 1080.0) as i32;
                                    let _ = enigo.move_mouse(dx_pix, dy_pix, Coordinate::Rel);
                                }
                            }
                        }
                    }

                    "drag_end" => {
                        // 拖拽结束：松开左键
                        let _ = enigo.button(Button::Left, Direction::Release);
                    }

                    "scroll" => {
                        // 滚动（这里不是真正的滚轮，而是模拟光标移动）
                        if let Some(delta) = json.get("delta") {
                            if let (Some(dx), Some(dy)) = (
                                delta.get("dx").and_then(|v| v.as_f64()),
                                delta.get("dy").and_then(|v| v.as_f64()),
                            ) {
                                if let Ok((w, h)) = enigo.main_display() {
                                    let dx_pix = (dx * w as f64) as i32;
                                    let dy_pix = (dy * h as f64) as i32;
                                    let _ = enigo.move_mouse(dx_pix, dy_pix, Coordinate::Rel);
                                } else {
                                    let dx_pix = (dx * 1920.0) as i32;
                                    let dy_pix = (dy * 1080.0) as i32;
                                    let _ = enigo.move_mouse(dx_pix, dy_pix, Coordinate::Rel);
                                }
                            }
                        }
                    }

                    other => {
                        log_println!("[decode_mouse_event] 未知 touchpad 事件: {}", other);
                    }
                }
            }
        }

        // —— 传统 mouse_xxx 事件 ——
        "mouse_move" => {
            if let (Some(rel_x), Some(rel_y)) = (
                json.get("x").and_then(|v| v.as_f64()),
                json.get("y").and_then(|v| v.as_f64()),
            ) {
                match enigo.main_display() {
                    Ok((w, h)) => {
                        let abs_x = (rel_x * w as f64) as i32;
                        let abs_y = (rel_y * h as f64) as i32;
                        let _ = enigo.move_mouse(abs_x, abs_y, Coordinate::Abs);
                    }
                    Err(_) => {
                        let abs_x = (rel_x * 1920.0) as i32;
                        let abs_y = (rel_y * 1080.0) as i32;
                        let _ = enigo.move_mouse(abs_x, abs_y, Coordinate::Abs);
                    }
                }
            }
        }

        "mouse_click" => {
            let is_double = json
                .get("double_tap")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let button_str = json
                .get("button")
                .and_then(|v| v.as_str())
                .unwrap_or("left");

            let btn = if is_double {
                Button::Right
            } else {
                match button_str {
                    "left" => Button::Left,
                    "right" => Button::Right,
                    "middle" => Button::Middle,
                    "back" => Button::Back,
                    "forward" => Button::Forward,
                    _ => Button::Left,
                }
            };
            let _ = enigo.button(btn, Direction::Click);
        }

        "mouse_down" => {
            let button_str = json
                .get("button")
                .and_then(|v| v.as_str())
                .unwrap_or("left");
            let btn = match button_str {
                "left" => Button::Left,
                "right" => Button::Right,
                "middle" => Button::Middle,
                "back" => Button::Back,
                "forward" => Button::Forward,
                _ => Button::Left,
            };
            let _ = enigo.button(btn, Direction::Press);
        }

        "mouse_up" => {
            let button_str = json
                .get("button")
                .and_then(|v| v.as_str())
                .unwrap_or("left");
            let btn = match button_str {
                "left" => Button::Left,
                "right" => Button::Right,
                "middle" => Button::Middle,
                "back" => Button::Back,
                "forward" => Button::Forward,
                _ => Button::Left,
            };
            let _ = enigo.button(btn, Direction::Release);
        }

        "scroll" => {
            // 与 touchpad scroll 一样，模拟光标移动
            if let Some(dy) = json.get("delta_y").and_then(|v| v.as_f64()) {
                if let Ok((w, h)) = enigo.main_display() {
                    let dx_pix = 0;
                    let dy_pix = (dy * h as f64) as i32;
                    let _ = enigo.move_mouse(dx_pix, dy_pix, Coordinate::Rel);
                } else {
                    let dy_pix = (dy * 1080.0) as i32;
                    let _ = enigo.move_mouse(0, dy_pix, Coordinate::Rel);
                }
            }
        }

        other => {
            log_println!("[decode_mouse_event] 未知鼠标命令: {}", other);
        }
    }
}

// 解码并执行键盘相关操作（包括组合键）
pub fn decode_keyboard_event(enigo: &mut Enigo, json: &Value) {
    let cmd_lower = json
        .get("cmd")
        .and_then(|v| v.as_str())
        .map(|s| s.to_lowercase())
        .or_else(|| {
            json.get("type")
                .and_then(|v| v.as_str())
                .map(|s| s.to_lowercase())
        });

    let cmd = match cmd_lower {
        Some(c) => c,
        None => return,
    };

    match cmd.as_str() {
        "key_down" | "keydown" => {
            if let Some(kstr) = json.get("key").and_then(|v| v.as_str()) {
                if let Some(k) = map_key(kstr) {
                    let _ = enigo.key(k, Direction::Press);
                } else {
                    log_println!("[decode_keyboard_event] 未知 key_down 键名: {}", kstr);
                }
            }
        }

        "key_up" | "keyup" => {
            if let Some(kstr) = json.get("key").and_then(|v| v.as_str()) {
                if let Some(k) = map_key(kstr) {
                    let _ = enigo.key(k, Direction::Release);
                } else {
                    log_println!("[decode_keyboard_event] 未知 key_up 键名: {}", kstr);
                }
            }
        }

        // 支持以下几种组合键标识：key_combo / keycombo / keycombine
        "key_combo" | "keycombo" | "keycombine" => {
            // 先尝试 json["keys"]，如果为空再尝试 json["key"]
            let arr_opt = json
                .get("keys")
                .and_then(|v| v.as_array())
                .or_else(|| json.get("key").and_then(|v| v.as_array()));

            if let Some(arr) = arr_opt {
                // 先按下所有组合键
                for item in arr {
                    if let Some(kstr) = item.as_str() {
                        if let Some(k) = map_key(kstr) {
                            let _ = enigo.key(k, Direction::Press);
                        } else {
                            log_println!("[decode_keyboard_event] 未知组合键 键名: {}", kstr);
                        }
                    }
                }
                // 再松开所有组合键
                for item in arr {
                    if let Some(kstr) = item.as_str() {
                        if let Some(k) = map_key(kstr) {
                            let _ = enigo.key(k, Direction::Release);
                        }
                    }
                }
            }
        }

        other => {
            log_println!("[decode_keyboard_event] 未知键盘命令: {}", other);
        }
    }
}

pub fn decode_joystick_event(enigo: &mut Enigo, json: &Value) {
    let cmd_lower = json
        .get("cmd")
        .and_then(|v| v.as_str())
        .map(|s| s.to_lowercase())
        .or_else(|| {
            json.get("type")
                .and_then(|v| v.as_str())
                .map(|s| s.to_lowercase())
        });

    let cmd = match cmd_lower {
        Some(c) => c,
        None => return,
    };
    match cmd.as_str() {
        "joystick" => {}
        other => {
            log_println!("[decode_keyboard_event] 未知键盘命令: {}", other);
        }
    }
}
// 主入口：根据 JSON 中的 “type” 或 “cmd” 字段，将事件分派到鼠标或键盘解码
pub fn decode_and_dispatch(enigo: &mut Enigo, json: &Value) {
    if let Some(typ) = json.get("type").and_then(|v| v.as_str()) {
        let lowercase = typ.to_lowercase();

        // Touchpad / Mouse 都交给鼠标解码
        if lowercase == "mouse" || lowercase == "touchpad" {
            decode_mouse_event(enigo, json);
        }
        // Keyboard
        else if lowercase == "keyboard" {
            decode_keyboard_event(enigo, json);
        }
        // 如果 type 本身就是 mouse_xxx
        else if lowercase.starts_with("mouse_") || lowercase == "touchpad" {
            decode_mouse_event(enigo, json);
        }
        // 如果 type 本身就是 keyxxx
        else if lowercase.starts_with("key") {
            decode_keyboard_event(enigo, json);
        }
        // Flutter 手势 “tapDown/tapUp”/“panStart/panUpdate/panEnd”
        else if lowercase == "tapdown" || lowercase == "tap_up" || lowercase == "tapup" {
            decode_mouse_event(enigo, json);
        } else if lowercase.starts_with("pan") {
            decode_mouse_event(enigo, json);
        } else {
            log_println!("[decode_and_dispatch] 未知 type: {}", typ);
        }
    } else if let Some(cmd) = json.get("cmd").and_then(|v| v.as_str()) {
        let lowercase = cmd.to_lowercase();
        if lowercase.starts_with("mouse_") || lowercase == "touchpad" {
            decode_mouse_event(enigo, json);
        } else if lowercase.starts_with("key") {
            decode_keyboard_event(enigo, json);
        } else {
            log_println!(
                "[decode_and_dispatch] 无法判定 cmd 属于 mouse 还是 keyboard: {}",
                cmd
            );
        }
    } else {
        log_println!(
            "[decode_and_dispatch] JSON 中既没有 type 也没有 cmd 字段: {}",
            json
        );
    }
}
