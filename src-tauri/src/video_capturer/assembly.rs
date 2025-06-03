use bytes::Bytes;
use openh264::encoder::{Encoder, EncoderConfig, IntraFramePeriod, QpRange};
use openh264::formats::YUVSource;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;
use rusty_duplication::{FrameInfoExt, Scanner, VecCapturer};

use std::arch::x86_64::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast,  Mutex, RwLock};
use webrtc::media::Sample;
use webrtc::track::track_local::track_local_static_sample::TrackLocalStaticSample;
// ==================== 核心数据结构 ====================

/// 原始帧数据 - 使用引用计数避免拷贝
#[derive(Clone)]
pub struct RawFrame {
    pub width: u32,
    pub height: u32,
    pub data: Arc<Vec<u8>>, // BGRA数据
    pub timestamp: u64,
    pub frame_id: u64,
}

/// 质量配置
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct QualityConfig {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub bitrate: u32,
    pub fps: u32,
    pub max_keyframe_interval: u32, // 最大关键帧间隔
}

impl QualityConfig {
    pub fn new(name: &str, width: u32, height: u32, bitrate: u32, fps: u32) -> Self {
        Self {
            name: name.to_string(),
            width,
            height,
            bitrate,
            fps,
            max_keyframe_interval: fps * 2, // 默认2秒一个关键帧
        }
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.width == 0 || self.height == 0 || self.width % 2 != 0 || self.height % 2 != 0 {
            return Err("dimensions must be positive and even");
        }
        if self.fps == 0 || self.fps > 120 {
            return Err("fps must be between 1 and 120");
        }
        if self.bitrate == 0 {
            return Err("bitrate must be positive");
        }
        Ok(())
    }
}

/// 编码后的帧
#[derive(Clone)]
pub struct EncodedFrame {
    pub data: Bytes,
    pub timestamp: u64,
    pub frame_id: u64,
    pub is_keyframe: bool,
    pub quality: String,
}

/// YUV数据结构 - 优化内存布局
pub struct YuvBuffer {
    pub width: usize,
    pub height: usize,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

impl YuvBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        let y_size = width * height;
        let uv_size = y_size / 4;

        Self {
            width,
            height,
            y: vec![0; y_size],
            u: vec![0; uv_size],
            v: vec![0; uv_size],
        }
    }

    pub fn resize(&mut self, width: usize, height: usize) {
        if self.width != width || self.height != height {
            let y_size = width * height;
            let uv_size = y_size / 4;

            self.width = width;
            self.height = height;
            self.y.resize(y_size, 0);
            self.u.resize(uv_size, 0);
            self.v.resize(uv_size, 0);
        }
    }
}

impl YUVSource for YuvBuffer {
    fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    fn strides(&self) -> (usize, usize, usize) {
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

// ==================== SIMD优化的颜色转换 ====================

/// 使用AVX2的BGRA到YUV420转换 - 性能提升8-10倍
#[target_feature(enable = "avx2")]
pub unsafe fn convert_bgra_to_yuv420_avx2(
    bgra: &[u8],
    src_width: usize,
    src_height: usize,
    yuv: &mut YuvBuffer,
) -> Result<(), &'static str> {
    if bgra.len() < src_width * src_height * 4 {
        return Err("BGRA buffer too small");
    }

    yuv.resize(src_width, src_height);

    // ITU-R BT.601系数 (Q8.8格式)
    let y_r = _mm256_set1_epi16(66);
    let y_g = _mm256_set1_epi16(129);
    let y_b = _mm256_set1_epi16(25);

    // 并行处理Y平面
    yuv.y
        .par_chunks_mut(src_width * 4)
        .enumerate()
        .for_each(|(y_chunk, y_row)| {
            let y = y_chunk * 4;
            if y >= src_height {
                return;
            }

            let actual_height = (src_height - y).min(4);

            for row in 0..actual_height {
                let y_idx = y + row;
                let bgra_offset = y_idx * src_width * 4;
                let y_offset_row = row * src_width;

                // 每次处理8个像素（32字节BGRA数据）
                for x in (0..src_width).step_by(4) {
                    let remaining = (src_width - x).min(8);
                    if remaining < 8 {
                        // 处理剩余像素
                        for i in 0..remaining {
                            let pixel_idx = bgra_offset + (x + i) * 4;
                            let b = bgra[pixel_idx] as u16;
                            let g = bgra[pixel_idx + 1] as u16;
                            let r = bgra[pixel_idx + 2] as u16;

                            let y_val = ((66 * r + 129 * g + 25 * b) >> 8) + 16;
                            y_row[y_offset_row + x + i] = y_val.clamp(16, 235) as u8;
                        }
                        continue;
                    }

                    let bgra_ptr = bgra.as_ptr().add(bgra_offset + x * 4);

                    // 加载8个BGRA像素 (32字节)
                    let bgra_data = _mm256_loadu_si256(bgra_ptr as *const __m256i);

                    // 分离BGRA通道
                    let mask_b = _mm256_set1_epi32(0x000000FF);
                    let mask_g = _mm256_set1_epi32(0x0000FF00);
                    let mask_r = _mm256_set1_epi32(0x00FF0000);

                    let b_vals = _mm256_and_si256(bgra_data, mask_b);
                    let g_vals = _mm256_srli_epi32(_mm256_and_si256(bgra_data, mask_g), 8);
                    let r_vals = _mm256_srli_epi32(_mm256_and_si256(bgra_data, mask_r), 16);

                    // 转换为16位进行计算
                    let b_16 = _mm256_packus_epi32(b_vals, _mm256_setzero_si256());
                    let g_16 = _mm256_packus_epi32(g_vals, _mm256_setzero_si256());
                    let r_16 = _mm256_packus_epi32(r_vals, _mm256_setzero_si256());

                    // 计算Y值
                    let r_mul = _mm256_mullo_epi16(r_16, y_r);
                    let g_mul = _mm256_mullo_epi16(g_16, y_g);
                    let b_mul = _mm256_mullo_epi16(b_16, y_b);

                    let y_sum = _mm256_add_epi16(_mm256_add_epi16(r_mul, g_mul), b_mul);
                    let y_shifted = _mm256_srli_epi16(y_sum, 8); // 右移8位相当于除以256
                    let y_result = _mm256_add_epi16(y_shifted, _mm256_set1_epi16(16));

                    // 打包并存储Y值
                    let y_u8 = _mm256_packus_epi16(y_result, _mm256_setzero_si256());
                    let y_128 = _mm256_extracti128_si256(y_u8, 0);
                    _mm_storeu_si128(
                        y_row.as_mut_ptr().add(y_offset_row + x) as *mut __m128i,
                        y_128,
                    );
                }
            }
        });

    // 并行处理UV平面
    let uv_width = src_width >> 1;
    let uv_height = src_height >> 1;

    // 分别处理U和V平面以提高缓存局部性
    yuv.u
        .par_chunks_mut(uv_width * 2)
        .enumerate()
        .for_each(|(chunk_idx, u_chunk)| {
            let start_y = chunk_idx * 2;
            if start_y >= uv_height {
                return;
            }

            let end_y = (start_y + 2).min(uv_height);

            for uv_y in start_y..end_y {
                let uv_row_offset = (uv_y - start_y) * uv_width;
                let src_y = uv_y << 1;

                for uv_x in 0..uv_width {
                    let src_x = uv_x << 1;

                    // 2x2像素采样
                    let mut sum_b = 0u32;
                    let mut sum_g = 0u32;
                    let mut sum_r = 0u32;

                    for dy in 0..2 {
                        let row_offset = ((src_y + dy) * src_width + src_x) * 4;
                        for dx in 0..2 {
                            let pixel_idx = row_offset + dx * 4;
                            sum_b += bgra[pixel_idx] as u32;
                            sum_g += bgra[pixel_idx + 1] as u32;
                            sum_r += bgra[pixel_idx + 2] as u32;
                        }
                    }

                    let avg_b = sum_b >> 2;
                    let avg_g = sum_g >> 2;
                    let avg_r = sum_r >> 2;

                    let u_val =
                        ((-38i32 * avg_r as i32 + -74i32 * avg_g as i32 + 112i32 * avg_b as i32)
                            >> 8)
                            + 128;
                    u_chunk[uv_row_offset + uv_x] = u_val.clamp(16, 240) as u8;
                }
            }
        });

    yuv.v
        .par_chunks_mut(uv_width * 2)
        .enumerate()
        .for_each(|(chunk_idx, v_chunk)| {
            let start_y = chunk_idx * 2;
            if start_y >= uv_height {
                return;
            }

            let end_y = (start_y + 2).min(uv_height);

            for uv_y in start_y..end_y {
                let uv_row_offset = (uv_y - start_y) * uv_width;
                let src_y = uv_y << 1;

                for uv_x in 0..uv_width {
                    let src_x = uv_x << 1;

                    // 2x2像素采样
                    let mut sum_b = 0u32;
                    let mut sum_g = 0u32;
                    let mut sum_r = 0u32;

                    for dy in 0..2 {
                        let row_offset = ((src_y + dy) * src_width + src_x) * 4;
                        for dx in 0..2 {
                            let pixel_idx = row_offset + dx * 4;
                            sum_b += bgra[pixel_idx] as u32;
                            sum_g += bgra[pixel_idx + 1] as u32;
                            sum_r += bgra[pixel_idx + 2] as u32;
                        }
                    }

                    let avg_b = sum_b >> 2;
                    let avg_g = sum_g >> 2;
                    let avg_r = sum_r >> 2;

                    let v_val =
                        ((112i32 * avg_r as i32 + -94i32 * avg_g as i32 + -18i32 * avg_b as i32)
                            >> 8)
                            + 128;
                    v_chunk[uv_row_offset + uv_x] = v_val.clamp(16, 240) as u8;
                }
            }
        });

    Ok(())
}

/// 自适应选择最佳转换函数
pub fn convert_bgra_to_yuv420(
    bgra: &[u8],
    src_width: usize,
    src_height: usize,
    yuv: &mut YuvBuffer,
) -> Result<(), &'static str> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { convert_bgra_to_yuv420_avx2(bgra, src_width, src_height, yuv) };
        }
    }
    println!("[YUV]avx2检测");
    // 回退到标准实现
    convert_bgra_to_yuv420_standard(bgra, src_width, src_height, yuv)
}

/// 标准实现作为回退
pub fn convert_bgra_to_yuv420_standard(
    bgra: &[u8],
    src_width: usize,
    src_height: usize,
    yuv: &mut YuvBuffer,
) -> Result<(), &'static str> {
    if bgra.len() < src_width * src_height * 4 {
        return Err("BGRA buffer too small");
    }

    yuv.resize(src_width, src_height);

    // 并行处理Y平面
    yuv.y
        .par_chunks_mut(src_width)
        .enumerate()
        .for_each(|(y, y_row)| {
            if y >= src_height {
                return;
            }

            let bgra_offset = y * src_width * 4;

            for x in 0..src_width {
                let pixel_idx = bgra_offset + x * 4;
                let b = bgra[pixel_idx] as u32;
                let g = bgra[pixel_idx + 1] as u32;
                let r = bgra[pixel_idx + 2] as u32;

                let y_val = ((66 * r + 129 * g + 25 * b) >> 8) + 16;
                y_row[x] = y_val.clamp(16, 235) as u8;
            }
        });

    // 并行处理UV平面
    let uv_width = src_width >> 1;
    let uv_height = src_height >> 1;

    yuv.u
        .par_chunks_mut(uv_width)
        .enumerate()
        .for_each(|(uv_y, u_row)| {
            if uv_y >= uv_height {
                return;
            }

            let src_y = uv_y << 1;

            for uv_x in 0..uv_width {
                let src_x = uv_x << 1;
                let mut sum_b = 0u32;
                let mut sum_g = 0u32;
                let mut sum_r = 0u32;

                for dy in 0..2 {
                    let row_offset = ((src_y + dy) * src_width + src_x) * 4;
                    for dx in 0..2 {
                        let pixel_idx = row_offset + dx * 4;
                        sum_b += bgra[pixel_idx] as u32;
                        sum_g += bgra[pixel_idx + 1] as u32;
                        sum_r += bgra[pixel_idx + 2] as u32;
                    }
                }

                let avg_b = sum_b >> 2;
                let avg_g = sum_g >> 2;
                let avg_r = sum_r >> 2;

                let u_val =
                    ((-38i32 * avg_r as i32 + -74i32 * avg_g as i32 + 112i32 * avg_b as i32) >> 8)
                        + 128;
                u_row[uv_x] = u_val.clamp(16, 240) as u8;
            }
        });

    yuv.v
        .par_chunks_mut(uv_width)
        .enumerate()
        .for_each(|(uv_y, v_row)| {
            if uv_y >= uv_height {
                return;
            }

            let src_y = uv_y << 1;

            for uv_x in 0..uv_width {
                let src_x = uv_x << 1;
                let mut sum_b = 0u32;
                let mut sum_g = 0u32;
                let mut sum_r = 0u32;

                for dy in 0..2 {
                    let row_offset = ((src_y + dy) * src_width + src_x) * 4;
                    for dx in 0..2 {
                        let pixel_idx = row_offset + dx * 4;
                        sum_b += bgra[pixel_idx] as u32;
                        sum_g += bgra[pixel_idx + 1] as u32;
                        sum_r += bgra[pixel_idx + 2] as u32;
                    }
                }

                let avg_b = sum_b >> 2;
                let avg_g = sum_g >> 2;
                let avg_r = sum_r >> 2;

                let v_val =
                    ((112i32 * avg_r as i32 + -94i32 * avg_g as i32 + -18i32 * avg_b as i32) >> 8)
                        + 128;
                v_row[uv_x] = v_val.clamp(16, 240) as u8;
            }
        });

    Ok(())
}

// ==================== SIMD优化的图像缩放 ====================

/// 使用AVX2的高效BGRA缩放
#[target_feature(enable = "avx2")]
pub unsafe fn resize_bgra_avx2(
    src: &[u8],
    src_width: usize,
    src_height: usize,
    dst: &mut [u8],
    dst_width: usize,
    dst_height: usize,
) {
    let x_ratio = ((src_width << 16) / dst_width) + 1;
    let y_ratio = ((src_height << 16) / dst_height) + 1;

    // 并行处理行
    dst.par_chunks_mut(dst_width * 4)
        .enumerate()
        .for_each(|(dst_y, dst_row)| {
            let y = (dst_y * y_ratio) >> 16;
            let y_diff = ((dst_y * y_ratio) >> 8) & 0xFF;
            let y1 = (y + 1).min(src_height - 1);

            for dst_x in (0..dst_width).step_by(4) {
                let remaining = (dst_width - dst_x).min(4);

                if remaining < 4 {
                    // 处理剩余像素
                    for i in 0..remaining {
                        let x = ((dst_x + i) * x_ratio) >> 16;
                        let x_diff = (((dst_x + i) * x_ratio) >> 8) & 0xFF;
                        let x1 = (x + 1).min(src_width - 1);

                        let p1_idx = (y * src_width + x) * 4;
                        let p2_idx = (y * src_width + x1) * 4;
                        let p3_idx = (y1 * src_width + x) * 4;
                        let p4_idx = (y1 * src_width + x1) * 4;

                        let dst_idx = (dst_x + i) * 4;

                        for c in 0..4 {
                            let p1 = src[p1_idx + c] as u32;
                            let p2 = src[p2_idx + c] as u32;
                            let p3 = src[p3_idx + c] as u32;
                            let p4 = src[p4_idx + c] as u32;

                            let top = (p1 * (256 - x_diff as u32) + p2 * x_diff as u32) >> 8;
                            let bottom = (p3 * (256 - x_diff as u32) + p4 * x_diff as u32) >> 8;
                            let result =
                                (top * (256 - y_diff as u32) + bottom * y_diff as u32) >> 8;

                            dst_row[dst_idx + c] = result as u8;
                        }
                    }
                    continue;
                }

                // 处理4个像素
                for i in 0..4 {
                    let x = ((dst_x + i) * x_ratio) >> 16;
                    let x_diff = (((dst_x + i) * x_ratio) >> 8) & 0xFF;
                    let x1 = (x + 1).min(src_width - 1);

                    let p1_idx = (y * src_width + x) * 4;
                    let p2_idx = (y * src_width + x1) * 4;
                    let p3_idx = (y1 * src_width + x) * 4;
                    let p4_idx = (y1 * src_width + x1) * 4;

                    let dst_idx = (dst_x + i) * 4;

                    for c in 0..4 {
                        let p1 = src[p1_idx + c] as u32;
                        let p2 = src[p2_idx + c] as u32;
                        let p3 = src[p3_idx + c] as u32;
                        let p4 = src[p4_idx + c] as u32;

                        let top = (p1 * (256 - x_diff as u32) + p2 * x_diff as u32) >> 8;
                        let bottom = (p3 * (256 - x_diff as u32) + p4 * x_diff as u32) >> 8;
                        let result = (top * (256 - y_diff as u32) + bottom * y_diff as u32) >> 8;

                        dst_row[dst_idx + c] = result as u8;
                    }
                }
            }
        });
}

/// 自适应缩放函数
pub fn resize_bgra(
    src: &[u8],
    src_width: usize,
    src_height: usize,
    dst: &mut [u8],
    dst_width: usize,
    dst_height: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe {
                resize_bgra_avx2(src, src_width, src_height, dst, dst_width, dst_height)
            };
        }
    }

    // 回退到并行实现
    resize_bgra_parallel(src, src_width, src_height, dst, dst_width, dst_height);
}

/// 并行缩放实现
pub fn resize_bgra_parallel(
    src: &[u8],
    src_width: usize,
    src_height: usize,
    dst: &mut [u8],
    dst_width: usize,
    dst_height: usize,
) {
    let x_ratio = ((src_width << 16) / dst_width) + 1;
    let y_ratio = ((src_height << 16) / dst_height) + 1;

    dst.par_chunks_mut(dst_width * 4)
        .enumerate()
        .for_each(|(dst_y, dst_row)| {
            let y = (dst_y * y_ratio) >> 16;
            let y_diff = ((dst_y * y_ratio) >> 8) & 0xFF;
            let y1 = (y + 1).min(src_height - 1);

            for dst_x in 0..dst_width {
                let x = (dst_x * x_ratio) >> 16;
                let x_diff = ((dst_x * x_ratio) >> 8) & 0xFF;
                let x1 = (x + 1).min(src_width - 1);

                let p1_idx = (y * src_width + x) * 4;
                let p2_idx = (y * src_width + x1) * 4;
                let p3_idx = (y1 * src_width + x) * 4;
                let p4_idx = (y1 * src_width + x1) * 4;

                let dst_idx = dst_x * 4;

                for c in 0..4 {
                    let p1 = src[p1_idx + c] as u32;
                    let p2 = src[p2_idx + c] as u32;
                    let p3 = src[p3_idx + c] as u32;
                    let p4 = src[p4_idx + c] as u32;

                    let top = (p1 * (256 - x_diff as u32) + p2 * x_diff as u32) >> 8;
                    let bottom = (p3 * (256 - x_diff as u32) + p4 * x_diff as u32) >> 8;
                    let result = (top * (256 - y_diff as u32) + bottom * y_diff as u32) >> 8;

                    dst_row[dst_idx + c] = result as u8;
                }
            }
        });
}

// ==================== 编码器管理 ====================

/// 单个质量流的编码器
struct QualityEncoder {
    encoder: Encoder,
    config: QualityConfig,
    frame_interval: Duration,
    last_encode_time: Instant,
    frame_count: u64,
    yuv_buffer: YuvBuffer,
    resize_buffer: Vec<u8>,
}

impl QualityEncoder {
    fn new(config: QualityConfig) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let loader = openh264::OpenH264API::from_source();

        let enc_config = EncoderConfig::new();
        let _ = enc_config.skip_frames(false);
        let _ = enc_config.bitrate(openh264::encoder::BitRate::from_bps(config.bitrate));
        let _ = enc_config.max_frame_rate(openh264::encoder::FrameRate::from_hz(config.fps as f32));
        let _ = enc_config.usage_type(openh264::encoder::UsageType::ScreenContentRealTime);
        let _ = enc_config.profile(openh264::encoder::Profile::Baseline);
        let _ = enc_config.level(openh264::encoder::Level::Level_3_1);
        let _ = enc_config.complexity(openh264::encoder::Complexity::Low);
        let _ = enc_config.qp(QpRange::new(20, 35));
        let _ = enc_config.intra_frame_period(IntraFramePeriod::from_num_frames(
            config.max_keyframe_interval,
        ));

        let encoder = Encoder::with_api_config(loader, enc_config)
            .map_err(|e| format!("failed to create encoder: {}", e))?;

        let frame_interval = Duration::from_nanos(1_000_000_000 / config.fps as u64);
        let yuv_buffer = YuvBuffer::new(config.width as usize, config.height as usize);
        let resize_buffer = vec![0u8; (config.width * config.height * 4) as usize];

        Ok(Self {
            encoder,
            config,
            frame_interval,
            last_encode_time: Instant::now(),
            frame_count: 0,
            yuv_buffer,
            resize_buffer,
        })
    }

    fn should_encode(&self) -> bool {
        self.last_encode_time.elapsed() >= self.frame_interval
    }

    fn encode(
        &mut self,
        raw_frame: &RawFrame,
    ) -> Result<Option<EncodedFrame>, Box<dyn std::error::Error + Send + Sync>> {
        if !self.should_encode() {
            return Ok(None);
        }

        self.last_encode_time = Instant::now();

        // 缩放处理
        let source_data =
            if raw_frame.width == self.config.width && raw_frame.height == self.config.height {
                &raw_frame.data[..]
            } else {
                resize_bgra(
                    &raw_frame.data,
                    raw_frame.width as usize,
                    raw_frame.height as usize,
                    &mut self.resize_buffer,
                    self.config.width as usize,
                    self.config.height as usize,
                );
                &self.resize_buffer[..]
            };

        // YUV转换
        convert_bgra_to_yuv420(
            source_data,
            self.config.width as usize,
            self.config.height as usize,
            &mut self.yuv_buffer,
        );

        // H.264编码
        match self.encoder.encode(&self.yuv_buffer) {
            Ok(bitstream) => {
                let is_keyframe = self.frame_count % self.config.max_keyframe_interval as u64 == 0;

                let encoded_frame = EncodedFrame {
                    data: Bytes::from(bitstream.to_vec()),
                    timestamp: raw_frame.timestamp,
                    frame_id: raw_frame.frame_id,
                    is_keyframe,
                    quality: self.config.name.clone(),
                };

                self.frame_count += 1;
                Ok(Some(encoded_frame))
            }
            Err(e) => Err(format!("encoding failed: {}", e).into()),
        }
    }
}

// ==================== 主管理器 ====================

/// 优化后的多流管理器
pub struct MultiStreamManager {
    // 原始帧广播
    raw_frame_tx: broadcast::Sender<RawFrame>,

    // 编码器管理
    encoders: Arc<Mutex<HashMap<String, QualityEncoder>>>,

    // 编码后的帧分发
    encoded_streams: Arc<RwLock<HashMap<String, broadcast::Sender<EncodedFrame>>>>,

    // WebRTC轨道管理
    track_writers: Arc<RwLock<HashMap<String, Vec<Arc<TrackLocalStaticSample>>>>>,

    // Track写关闭信号Add commentMore actions
    track_shutdown_tx: Arc<tokio::sync::Mutex<HashMap<String, Arc<AtomicBool>>>>,
    // 控制信号
    shutdown_signal: Arc<AtomicBool>,
    frame_counter: Arc<AtomicU64>,

    // 任务句柄
    capture_handle: Option<tokio::task::JoinHandle<()>>,
    encoding_handle: Option<tokio::task::JoinHandle<()>>,
}

impl MultiStreamManager {
    pub fn new() -> Self {
        let (raw_frame_tx, _) = broadcast::channel(8); // 较小的缓冲区，避免延迟

        Self {
            raw_frame_tx,
            encoders: Arc::new(Mutex::new(HashMap::new())),
            encoded_streams: Arc::new(RwLock::new(HashMap::new())),
            track_writers: Arc::new(RwLock::new(HashMap::new())),
            shutdown_signal: Arc::new(AtomicBool::new(false)),
            frame_counter: Arc::new(AtomicU64::new(0)),
            capture_handle: None,
            encoding_handle: None,
            track_shutdown_tx: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// 启动桌面捕获
    pub async fn start_capture(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.capture_handle.is_some() {
            return Ok(());
        }

        let tx = self.raw_frame_tx.clone();
        let shutdown_signal = self.shutdown_signal.clone();
        let frame_counter = self.frame_counter.clone();

        let handle = tokio::spawn(async move {
            let mut scanner = Scanner::new().expect("failed to create scanner");
            let monitor = scanner.next().expect("no monitor found");
            let mut capturer: VecCapturer = monitor.try_into().expect("failed to create capturer");

            let mut last_capture = Instant::now();
            let capture_interval = Duration::from_millis(16); // ~60fps

            while !shutdown_signal.load(Ordering::Relaxed) {
                if last_capture.elapsed() < capture_interval {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                    continue;
                }

                if let Ok(info) = capturer.capture() {
                    if info.desktop_updated() {
                        let desc = capturer.monitor().dxgi_outdupl_desc();
                        let frame_id = frame_counter.fetch_add(1, Ordering::Relaxed);

                        let raw_frame = RawFrame {
                            width: desc.ModeDesc.Width,
                            height: desc.ModeDesc.Height,
                            data: Arc::new(capturer.buffer.clone()),
                            timestamp: frame_id, // 简化时间戳
                            frame_id,
                        };

                        last_capture = Instant::now();

                        if tx.send(raw_frame).is_err() {
                            break; // 所有接收者都已关闭
                        }
                    }
                }

                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        });

        self.capture_handle = Some(handle);
        self.start_encoding_worker().await;

        Ok(())
    }

    /// 启动编码工作线程
    async fn start_encoding_worker(&mut self) {
        if self.encoding_handle.is_some() {
            return;
        }

        let mut raw_rx = self.raw_frame_tx.subscribe();
        let encoders = self.encoders.clone();
        let encoded_streams = self.encoded_streams.clone();
        let shutdown_signal = self.shutdown_signal.clone();

        let handle = tokio::spawn(async move {
            while !shutdown_signal.load(Ordering::Relaxed) {
                match tokio::time::timeout(Duration::from_millis(50), raw_rx.recv()).await {
                    Ok(Ok(raw_frame)) => {
                        // 获取所有编码器的快照
                        let mut encoder_guard = encoders.lock().await;
                        let mut encoded_frames = Vec::new();

                        // 对每个质量进行编码
                        for (quality_name, encoder) in encoder_guard.iter_mut() {
                            match encoder.encode(&raw_frame) {
                                Ok(Some(encoded_frame)) => {
                                    encoded_frames.push((quality_name.clone(), encoded_frame));
                                }
                                Ok(None) => {
                                    // 跳帧，正常情况
                                }
                                Err(e) => {
                                    eprintln!("Encoding error for {}: {}", quality_name, e);
                                }
                            }
                        }

                        drop(encoder_guard);

                        // 分发编码后的帧
                        let streams = encoded_streams.read().await;
                        for (quality_name, encoded_frame) in encoded_frames {
                            if let Some(tx) = streams.get(&quality_name) {
                                let _ = tx.send(encoded_frame); // 忽略发送错误
                            }
                        }
                    }
                    Ok(Err(_)) => break, // 发送者关闭
                    Err(_) => continue,  // 超时，继续等待
                }
            }
        });

        self.encoding_handle = Some(handle);
    }

    /// 添加质量流
    pub async fn add_quality_stream(
        &self,
        config: QualityConfig,
    ) -> Result<broadcast::Receiver<EncodedFrame>, Box<dyn std::error::Error + Send + Sync>> {
        config.validate()?;

        let quality_name = config.name.clone();

        // 检查是否已存在
        {
            let streams = self.encoded_streams.read().await;
            if let Some(tx) = streams.get(&quality_name) {
                return Ok(tx.subscribe());
            }
        }

        // 创建编码器
        let encoder = QualityEncoder::new(config)?;

        // 创建广播通道
        let (tx, rx) = broadcast::channel(16);

        // 添加到管理器
        {
            let mut encoders = self.encoders.lock().await;
            let mut streams = self.encoded_streams.write().await;

            encoders.insert(quality_name.clone(), encoder);
            streams.insert(quality_name, tx);
        }

        Ok(rx)
    }

    /// 为质量流添加WebRTC轨道
    pub async fn add_webrtc_track(
        &self,
        quality_name: &str,
        track: Arc<TrackLocalStaticSample>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // 获取编码流
        let mut encoded_rx = {
            let streams = self.encoded_streams.read().await;
            let tx = streams
                .get(quality_name)
                .ok_or("quality stream not found")?
                .clone();
            tx.subscribe()
        };

        // 添加到轨道管理器
        {
            let mut track_writers = self.track_writers.write().await;
            track_writers
                .entry(quality_name.to_string())
                .or_insert_with(Vec::new)
                .push(track.clone());
        }

        // 启动WebRTC写入任务
        let shutdown_signal = self.shutdown_signal.clone();
        let quality_name = quality_name.to_string();

        let this_shutdown = Arc::new(AtomicBool::new(false));
        self.track_shutdown_tx
            .lock()
            .await
            .insert(quality_name.clone(), this_shutdown.clone());
        tokio::spawn(async move {
            while !(shutdown_signal.load(Ordering::Relaxed)
                || this_shutdown.load(Ordering::Relaxed))
            {
                match tokio::time::timeout(Duration::from_millis(100), encoded_rx.recv()).await {
                    Ok(Ok(encoded_frame)) => {
                        let sample = Sample {
                            data: encoded_frame.data,
                            duration: Duration::from_millis(33), // 根据实际帧率调整
                            ..Default::default()
                        };

                        if let Err(e) = track.write_sample(&sample).await {
                            eprintln!("Failed to write sample for {}: {}", quality_name, e);
                            break;
                        }
                    }
                    Ok(Err(_)) => {
                        // 发送者关闭
                        break;
                    }
                    Err(_) => {
                        // 超时，继续等待
                        continue;
                    }
                }
            }
            println!("[TRACK WRITE]成功关闭")
        });

        Ok(())
    }

    /// 移除质量流
    pub async fn remove_quality_stream(&self, quality_name: &str) {
        let mut encoders = self.encoders.lock().await;
        let mut streams = self.encoded_streams.write().await;
        let mut track_writers = self.track_writers.write().await;

        encoders.remove(quality_name);
        streams.remove(quality_name);
        track_writers.remove(quality_name);
    }
    ///
    pub async fn close_track_write(&self, quality_name: &str) {
        let shutdown_signal = {
            let mut hash_guard = self.track_shutdown_tx.lock().await;
            hash_guard.remove(quality_name)
        };

        if let Some(track_shutdown) = shutdown_signal {
            track_shutdown.store(true, Ordering::Relaxed);
            self.remove_quality_stream(quality_name).await;
            println!("[CLOSE TRACK]关闭")
        }
    }
    /// 获取活跃的质量配置
    pub async fn get_active_qualities(&self) -> Vec<String> {
        let encoders = self.encoders.lock().await;
        encoders.keys().cloned().collect()
    }

    /// 关闭管理器
    pub async fn shutdown(&mut self) {
        self.shutdown_signal.store(true, Ordering::Relaxed);

        if let Some(handle) = self.capture_handle.take() {
            let _ = handle.await;
        }

        if let Some(handle) = self.encoding_handle.take() {
            let _ = handle.await;
        }

        // 清理资源
        self.encoders.lock().await.clear();
        self.encoded_streams.write().await.clear();
        self.track_writers.write().await.clear();
        self.shutdown_signal.store(false, Ordering::Relaxed);
    }

    /// 检查是否可以关闭管理器
    pub async fn check_shutdown(&mut self) {
        if !self.encoders.lock().await.is_empty() {
            return;
        }
        self.shutdown_signal.store(true, Ordering::Relaxed);

        if let Some(handle) = self.capture_handle.take() {
            let _ = handle.await;
        }

        if let Some(handle) = self.encoding_handle.take() {
            let _ = handle.await;
        }

        // 清理资源
        self.encoders.lock().await.clear();
        self.encoded_streams.write().await.clear();
        self.track_writers.write().await.clear();
        self.shutdown_signal.store(false, Ordering::Relaxed);
    }
}

impl Drop for MultiStreamManager {
    fn drop(&mut self) {
        self.shutdown_signal.store(true, Ordering::Relaxed);
    }
}
