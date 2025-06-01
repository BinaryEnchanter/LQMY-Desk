use openh264::formats::YUVSource;
// YUV 数据结构
pub struct YuvData {
    pub width: usize,
    pub height: usize,
    /// Y 平面，长度 = width * height
    pub y: Vec<u8>,
    /// U 平面，长度 = (width/2) * (height/2)
    pub u: Vec<u8>,
    /// V 平面，长度 = (width/2) * (height/2)
    pub v: Vec<u8>,
}

impl YuvData {
    /// 构造函数，从 BGRA 或者其他源转换后填充 y,u,v
    pub fn new(width: usize, height: usize, y: Vec<u8>, u: Vec<u8>, v: Vec<u8>) -> Self {
        assert_eq!(y.len(), width * height);
        assert_eq!(u.len(), (width / 2) * (height / 2));
        assert_eq!(v.len(), (width / 2) * (height / 2));
        Self {
            width,
            height,
            y,
            u,
            v,
        }
    }
}

impl YUVSource for YuvData {
    fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    fn strides(&self) -> (usize, usize, usize) {
        // Y 平面每行 stride = width
        // U、V 平面每行 stride = width/2
        (self.width, self.width / 2, self.width / 2)
    }

    fn y(&self) -> &[u8] {
        &self.y
    }

    fn u(&self) -> &[u8] {
        &self.u
    }

    fn v(&self) -> &[u8] {
        &self.v
    }
}
// 将 BGRA 转换为 YUV420
use rayon::prelude::*;

/// 超高性能 BGRA → YUV420
///
/// - BT.601 全范围，定点系数 8-bit frac  
///   Y  = ( 66*R + 129*G +  25*B + 128) >> 8  +  16  
///   U  = (-38*R -  74*G + 112*B + 128) >> 8  + 128  
///   V  = (112*R -  94*G -  18*B + 128) >> 8  + 128  
pub fn bgra_to_yuv420(bgra: &[u8], width: usize, height: usize) -> YuvData {
    let num_pixels = width * height;
    // 1. 预分配各平面
    let mut y_plane = vec![0u8; num_pixels];
    let mut u_plane = vec![0u8; num_pixels / 4];
    let mut v_plane = vec![0u8; num_pixels / 4];

    // 2. 定点系数 & 偏移
    const C_YR: i32 = 66;
    const C_YG: i32 = 129;
    const C_YB: i32 = 25;
    const C_UR: i32 = -38;
    const C_UG: i32 = -74;
    const C_UB: i32 = 112;
    const C_VR: i32 = 112;
    const C_VG: i32 = -94;
    const C_VB: i32 = -18;
    const OFF_Y: i32 = 16;
    const OFF_UV: i32 = 128;
    const RND: i32 = 128;

    // 3. 并行计算 Y
    y_plane
        .par_chunks_exact_mut(1024)
        .enumerate()
        .for_each(|(chunk_i, chunk)| {
            let base_idx = chunk_i * 1024;
            for (i, yv) in chunk.iter_mut().enumerate() {
                let idx = base_idx + i;
                let pix = unsafe { *(bgra.as_ptr().add(idx * 4) as *const u32) };
                let b = (pix & 0xFF) as i32;
                let g = ((pix >> 8) & 0xFF) as i32;
                let r = ((pix >> 16) & 0xFF) as i32;
                let y = ((C_YR * r + C_YG * g + C_YB * b + RND) >> 8) + OFF_Y;
                *yv = y.clamp(0, 255) as u8;
            }
        });

    // 4. 并行计算 UV，每行 blocks_per_row 个 2×2 块
    let blocks_per_row = width / 2;
    u_plane
        .par_chunks_exact_mut(blocks_per_row)
        .zip(v_plane.par_chunks_exact_mut(blocks_per_row))
        .enumerate()
        .for_each(|(by, (u_row, v_row))| {
            let y0 = by * 2;
            let y1 = y0 + 1;
            for bx in 0..blocks_per_row {
                let mut sum_u = 0i32;
                let mut sum_v = 0i32;
                // 2×2 像素
                for (yy, _y_off) in [(y0, y0), (y1, y1)] {
                    let base = (yy * width + bx * 2) * 4;
                    for dx in 0..2 {
                        let pix = unsafe { *(bgra.as_ptr().add(base + dx * 4) as *const u32) };
                        let b = (pix & 0xFF) as i32;
                        let g = ((pix >> 8) & 0xFF) as i32;
                        let r = ((pix >> 16) & 0xFF) as i32;
                        let u = ((C_UR * r + C_UG * g + C_UB * b + RND) >> 8) + OFF_UV;
                        let v = ((C_VR * r + C_VG * g + C_VB * b + RND) >> 8) + OFF_UV;
                        sum_u += u;
                        sum_v += v;
                    }
                }
                // 平均并写入
                let u_avg = (sum_u >> 2).clamp(0, 255) as u8;
                let v_avg = (sum_v >> 2).clamp(0, 255) as u8;
                u_row[bx] = u_avg;
                v_row[bx] = v_avg;
            }
        });

    YuvData {
        width,
        height,
        y: y_plane,
        u: u_plane,
        v: v_plane,
    }
}
// // 使用示例
// pub async fn example_usage() {
//     // 创建多流管理器
//     let manager = MultiStreamManager::new(50);

//     // 启动桌面捕获
//     manager.start_capture();

//     // 定义不同的质量档次
//     let qualities = vec![
//         QualityConfig {
//             name: "1080p".to_string(),
//             width: 1920,
//             height: 1080,
//             bitrate: 2000000,
//             fps: 30,
//         },
//         QualityConfig {
//             name: "720p".to_string(),
//             width: 1280,
//             height: 720,
//             bitrate: 1000000,
//             fps: 30,
//         },
//         QualityConfig {
//             name: "480p".to_string(),
//             width: 854,
//             height: 480,
//             bitrate: 500000,
//             fps: 24,
//         },
//     ];

//     // 为每个质量档次创建编码流
//     for quality in qualities {
//         let _encoded_rx = manager.add_quality_stream(quality).await;
//         // 现在可以将 encoded_rx 用于多个 WebRTC connections
//     }

//     // 当有新的 PeerConnection 时，为其创建对应质量的 track writer
//     // let track = Arc::new(TrackLocalStaticSample::new(...));
//     // manager.create_track_writer("720p", track).await;
// }
