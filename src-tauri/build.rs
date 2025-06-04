fn main() {
    tauri_build::build();
    println!("OUT_DIR = {}", std::env::var("OUT_DIR").unwrap());
}
