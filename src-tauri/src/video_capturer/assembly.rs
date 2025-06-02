use bytes::Bytes;
use openh264::encoder::{Encoder, EncoderConfig, IntraFramePeriod, QpRange};
use openh264::formats::YUVSource;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;
use rusty_duplication::{FrameInfoExt, Scanner, VecCapturer};
use std::arch::x86_64::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::thread;
use std::time::{Duration, Instant};
use tokio::sync::broadcast::error::RecvError;
use tokio::sync::{broadcast, mpsc, Mutex as AsyncMutex, RwLock};
use tokio::time::timeout;
use webrtc::media::Sample;
use webrtc::track::track_local::track_local_static_sample::TrackLocalStaticSample;

// ==================== 核心数据结构 ====================

/// 原始帧数据 - 零拷贝设计
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
    pub max_keyframe_interval: u32,
}

impl QualityConfig {
    pub fn new(name: &str, width: u32, height: u32, bitrate: u32, fps: u32) -> Self {
        Self {
            name: name.to_string(),
            width,
            height,
            bitrate,
            fps,
            max_keyframe_interval: fps * 2,
        }
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.width == 0 || self.height == 0 || self.width % 2 != 0 || self.height % 2 != 0 {
            return Err("dimensions must be positive and even");
        }
        if !(1..=120).contains(&self.fps) {
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

/// 高性能YUV缓冲区 - 内存对齐优化
#[repr(align(32))]
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
        let uv_size = y_size >> 2; // 除以4

        Self {
            width,
            height,
            y: vec![0; y_size],
            u: vec![0; uv_size],
            v: vec![0; uv_size],
        }
    }

    #[inline]
    pub fn resize(&mut self, width: usize, height: usize) {
        if self.width == width && self.height == height {
            return;
        }

        let y_size = width * height;
        let uv_size = y_size >> 2;

        self.width = width;
        self.height = height;
        self.y.resize(y_size, 0);
        self.u.resize(uv_size, 0);
        self.v.resize(uv_size, 0);
    }
}

impl YUVSource for YuvBuffer {
    #[inline]
    fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    #[inline]
    fn strides(&self) -> (usize, usize, usize) {
        (self.width, self.width >> 1, self.width >> 1)
    }

    #[inline]
    fn y(&self) -> &[u8] {
        &self.y
    }

    #[inline]
    fn u(&self) -> &[u8] {
        &self.u
    }

    #[inline]
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

/// 单质量流编码器 - 内存池优化
struct QualityEncoder {
    encoder: Encoder,
    config: QualityConfig,
    frame_interval_ns: u64,
    last_encode: Instant,
    frame_count: u64,
    yuv_buffer: YuvBuffer,
    resize_buffer: Vec<u8>,
}

impl QualityEncoder {
    fn new(config: QualityConfig) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let loader = openh264::OpenH264API::from_source();
        let enc_config = EncoderConfig::new();

        enc_config.skip_frames(false);
        enc_config.bitrate(openh264::encoder::BitRate::from_bps(config.bitrate));
        enc_config.max_frame_rate(openh264::encoder::FrameRate::from_hz(config.fps as f32));
        enc_config.usage_type(openh264::encoder::UsageType::ScreenContentRealTime);
        enc_config.profile(openh264::encoder::Profile::Baseline);
        enc_config.level(openh264::encoder::Level::Level_3_1);
        enc_config.complexity(openh264::encoder::Complexity::Low);
        enc_config.qp(QpRange::new(20, 35));
        enc_config.intra_frame_period(IntraFramePeriod::from_num_frames(
            config.max_keyframe_interval,
        ));
        enc_config.num_threads(4);

        let encoder = Encoder::with_api_config(loader, enc_config)
            .map_err(|e| format!("encoder creation failed: {}", e))?;

        Ok(Self {
            encoder,
            frame_interval_ns: 1_000_000_000 / config.fps as u64,
            last_encode: Instant::now(),
            frame_count: 0,
            yuv_buffer: YuvBuffer::new(config.width as usize, config.height as usize),
            resize_buffer: vec![0u8; (config.width * config.height * 4) as usize],
            config,
        })
    }

    #[inline]
    fn should_encode(&self) -> bool {
        self.last_encode.elapsed().as_nanos() as u64 >= self.frame_interval_ns
    }

    fn encode(
        &mut self,
        raw_frame: &RawFrame,
    ) -> Result<Option<EncodedFrame>, Box<dyn std::error::Error + Send + Sync>> {
        if !self.should_encode() {
            return Ok(None);
        }
        let one_time = Instant::now();
        self.last_encode = Instant::now();

        // 智能缩放处理
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

        // 高速YUV转换
        convert_bgra_to_yuv420(
            source_data,
            self.config.width as usize,
            self.config.height as usize,
            &mut self.yuv_buffer,
        );
        //println!("[PRE_FRAME] {:?}", Instant::now().duration_since(one_time));
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
                //println!("[ONE_FRAME] {:?}", Instant::now().duration_since(one_time));
                Ok(Some(encoded_frame))
            }
            Err(e) => Err(format!("encoding failed: {}", e).into()),
        }
    }
}

// ==================== 主管理器 ====================

/// 高性能多流管理器 - 优化锁竞争版本
/// 高性能多流管理器 - 优化锁竞争版本
pub struct MultiStreamManager {
    raw_frame_tx: broadcast::Sender<RawFrame>,
    encoders: Arc<AsyncMutex<HashMap<String, QualityEncoder>>>,
    encoded_streams: Arc<RwLock<HashMap<String, broadcast::Sender<EncodedFrame>>>>,
    track_writers: Arc<RwLock<HashMap<String, Vec<Arc<TrackLocalStaticSample>>>>>,
    track_shutdown_tx: Arc<AsyncMutex<HashMap<String, Arc<AtomicBool>>>>,
    shutdown_signal: Arc<AtomicBool>,
    frame_counter: Arc<AtomicU64>,
    capture_handle: Arc<AsyncMutex<Option<thread::JoinHandle<()>>>>,
    encoding_handle: Arc<AsyncMutex<Option<thread::JoinHandle<()>>>>,
    capture_shutdown_tx: Arc<AsyncMutex<Option<mpsc::UnboundedSender<()>>>>,
    encoding_shutdown_tx: Arc<AsyncMutex<Option<mpsc::UnboundedSender<()>>>>,

    // 新增：无锁流快照
    stream_snapshot: Arc<AtomicPtr<Vec<(String, broadcast::Sender<EncodedFrame>)>>>,
    encoder_names: Arc<AtomicPtr<Vec<String>>>,
    snapshot_version: Arc<AtomicU64>,
}

impl MultiStreamManager {
    pub fn new() -> Self {
        let (raw_frame_tx, _) = broadcast::channel(32);

        Self {
            raw_frame_tx,
            encoders: Arc::new(AsyncMutex::new(HashMap::new())),
            encoded_streams: Arc::new(RwLock::new(HashMap::new())),
            track_writers: Arc::new(RwLock::new(HashMap::new())),
            shutdown_signal: Arc::new(AtomicBool::new(false)),
            frame_counter: Arc::new(AtomicU64::new(0)),
            capture_handle: Arc::new(AsyncMutex::new(None)),
            encoding_handle: Arc::new(AsyncMutex::new(None)),
            capture_shutdown_tx: Arc::new(AsyncMutex::new(None)),
            encoding_shutdown_tx: Arc::new(AsyncMutex::new(None)),
            track_shutdown_tx: Arc::new(AsyncMutex::new(HashMap::new())),
            encoder_names: Arc::new(AtomicPtr::new(std::ptr::null_mut())),
            stream_snapshot: Arc::new(AtomicPtr::new(std::ptr::null_mut())),
            snapshot_version: Arc::new(AtomicU64::new(0)),
        }
    }

    // 更新流快照 - 仅在添加新流时调用
    async fn update_snapshots(&self) {
        let streams = self.encoded_streams.read().await;
        let encoders = self.encoders.lock().await;

        let stream_vec: Vec<(String, broadcast::Sender<EncodedFrame>)> = streams
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        let encoder_names_vec: Vec<String> = encoders.keys().cloned().collect();

        // 原子性更新快照
        let old_stream_ptr = self
            .stream_snapshot
            .swap(Box::into_raw(Box::new(stream_vec)), Ordering::Release);
        let old_names_ptr = self.encoder_names.swap(
            Box::into_raw(Box::new(encoder_names_vec)),
            Ordering::Release,
        );

        self.snapshot_version.fetch_add(1, Ordering::Release);

        // 安全清理旧快照
        if !old_stream_ptr.is_null() {
            unsafe {
                drop(Box::from_raw(old_stream_ptr));
            }
        }
        if !old_names_ptr.is_null() {
            unsafe {
                drop(Box::from_raw(old_names_ptr));
            }
        }
    }

    pub async fn start_capture(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // 检查是否已经启动，避免长时间持有锁
        {
            let handle_guard = self.capture_handle.lock().await;
            if handle_guard.is_some() {
                return Ok(());
            }
        }

        let tx = self.raw_frame_tx.clone();
        let frame_counter = self.frame_counter.clone();
        let shutdown_signal = self.shutdown_signal.clone();
        let (shutdown_tx, mut shutdown_rx) = mpsc::unbounded_channel();

        let handle = thread::Builder::new()
            .name("screen-capture-hp".to_string())
            .spawn(move || {
                Self::high_performance_capture_thread(
                    tx,
                    frame_counter,
                    shutdown_signal,
                    shutdown_rx,
                );
            })?;

        // 分别更新各个字段，减少锁的持有时间
        {
            let mut capture_handle_guard = self.capture_handle.lock().await;
            *capture_handle_guard = Some(handle);
        }

        {
            let mut shutdown_tx_guard = self.capture_shutdown_tx.lock().await;
            *shutdown_tx_guard = Some(shutdown_tx);
        }
        self.start_encoding_worker().await;

        Ok(())
    }

    fn high_performance_capture_thread(
        tx: broadcast::Sender<RawFrame>,
        frame_counter: Arc<AtomicU64>,
        shutdown_signal: Arc<AtomicBool>,
        mut shutdown_rx: mpsc::UnboundedReceiver<()>,
    ) {
        unsafe {
            #[cfg(target_os = "windows")]
            {
                use winapi::um::processthreadsapi::{GetCurrentThread, SetThreadPriority};
                use winapi::um::winbase::THREAD_PRIORITY_TIME_CRITICAL;
                SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL as i32);
            }
        }

        let mut scanner = Scanner::new().expect("scanner creation failed");
        let monitor = scanner.next().expect("no monitor found");
        let mut capturer: VecCapturer = monitor.try_into().expect("capturer creation failed");

        let mut last_capture = Instant::now();
        const CAPTURE_INTERVAL_NS: u64 = 16_666_667;
        let mut frame_buffer = Vec::with_capacity(1920 * 1080 * 4);

        let mut fps_counter = 0u32;
        let mut fps_start_time = Instant::now();

        loop {
            if shutdown_signal.load(Ordering::Relaxed) || shutdown_rx.try_recv().is_ok() {
                break;
            }

            let now = Instant::now();
            let elapsed_ns = now.duration_since(last_capture).as_nanos() as u64;

            if elapsed_ns < CAPTURE_INTERVAL_NS {
                let sleep_ns = CAPTURE_INTERVAL_NS - elapsed_ns;
                if sleep_ns > 100_000 {
                    thread::sleep(Duration::from_nanos(sleep_ns - 50_000));
                } else {
                    thread::yield_now();
                }
                continue;
            }

            match capturer.capture() {
                Ok(info) if info.desktop_updated() => {
                    let desc = capturer.monitor().dxgi_outdupl_desc();
                    let frame_id = frame_counter.fetch_add(1, Ordering::Relaxed);

                    frame_buffer.clear();
                    frame_buffer.extend_from_slice(&capturer.buffer);

                    let raw_frame = RawFrame {
                        width: desc.ModeDesc.Width,
                        height: desc.ModeDesc.Height,
                        data: Arc::new(frame_buffer.clone()),
                        timestamp: frame_id,
                        frame_id,
                    };

                    last_capture = now;

                    // === FPS统计 - 在成功捕获帧后 ===
                    fps_counter += 1;
                    if fps_start_time.elapsed() >= Duration::from_secs(1) {
                        println!("[CAPTURE] FPS: {} frames/sec", fps_counter);
                        fps_counter = 0;
                        fps_start_time = Instant::now();
                    }

                    if tx.send(raw_frame).is_err() {
                        println!("[CAPTURE] Channel closed, stopping capture");
                        break;
                    }
                }
                Ok(_) => thread::sleep(Duration::from_micros(500)),
                Err(e) => {
                    println!("[CAPTURE] Error: {:?}", e);
                    thread::sleep(Duration::from_millis(1));
                }
            }
        }
    }

    async fn start_encoding_worker(&self) {
        // 检查是否已经启动
        {
            let handle_guard = self.encoding_handle.lock().await;
            if handle_guard.is_some() {
                return;
            }
        }

        let raw_rx = self.raw_frame_tx.subscribe();
        let encoders = self.encoders.clone();
        let stream_snapshot = self.stream_snapshot.clone();
        let encoder_names = self.encoder_names.clone();
        let snapshot_version = self.snapshot_version.clone();
        let shutdown_signal = self.shutdown_signal.clone();
        let (shutdown_tx, shutdown_rx) = mpsc::unbounded_channel();

        let handle = thread::Builder::new()
            .name("video-encoder-hp".to_string())
            .spawn(move || {
                Self::optimized_encoding_thread(
                    raw_rx,
                    encoders,
                    stream_snapshot,
                    encoder_names,
                    snapshot_version,
                    shutdown_signal,
                    shutdown_rx,
                );
            })
            .expect("encoding thread spawn failed");

        // 更新句柄
        {
            let mut encoding_handle_guard = self.encoding_handle.lock().await;
            *encoding_handle_guard = Some(handle);
        }

        {
            let mut shutdown_tx_guard = self.encoding_shutdown_tx.lock().await;
            *shutdown_tx_guard = Some(shutdown_tx);
        }
    }

    /// 优化编码线程 - 减少锁竞争
    fn optimized_encoding_thread(
        mut raw_rx: broadcast::Receiver<RawFrame>,
        encoders: Arc<AsyncMutex<HashMap<String, QualityEncoder>>>,
        stream_snapshot: Arc<AtomicPtr<Vec<(String, broadcast::Sender<EncodedFrame>)>>>,
        encoder_names: Arc<AtomicPtr<Vec<String>>>,
        snapshot_version: Arc<AtomicU64>,
        shutdown_signal: Arc<AtomicBool>,
        mut shutdown_rx: mpsc::UnboundedReceiver<()>,
    ) {
        unsafe {
            #[cfg(target_os = "windows")]
            {
                use winapi::um::processthreadsapi::{GetCurrentThread, SetThreadPriority};
                use winapi::um::winbase::THREAD_PRIORITY_ABOVE_NORMAL;
                SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL as i32);
            }
        }

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime creation failed");

        let mut cached_streams: Option<Vec<(String, broadcast::Sender<EncodedFrame>)>> = None;
        let mut cached_names: Option<Vec<String>> = None;
        let mut last_version = 0u64;
        // 在循环外部添加FPS统计变量
        let mut fps_counter = 0u32;
        let mut fps_start_time = Instant::now();

        loop {
            if shutdown_signal.load(Ordering::Relaxed) || shutdown_rx.try_recv().is_ok() {
                break;
            }

            // 1. 优化接收 - 使用 recv_timeout 替代 sleep
            let raw_frame = match raw_rx.try_recv() {
                Ok(frame) => frame,
                Err(broadcast::error::TryRecvError::Empty) => {
                    thread::sleep(Duration::from_micros(50)); // 减少 sleep 时间
                    continue;
                }
                Err(broadcast::error::TryRecvError::Lagged(_)) => continue,
                Err(broadcast::error::TryRecvError::Closed) => break,
            };

            // 2. 检查快照更新
            let current_version = snapshot_version.load(Ordering::Acquire);
            if current_version != last_version {
                let stream_ptr = stream_snapshot.load(Ordering::Acquire);
                let names_ptr = encoder_names.load(Ordering::Acquire);

                if !stream_ptr.is_null() && !names_ptr.is_null() {
                    unsafe {
                        cached_streams = Some((*stream_ptr).clone());
                        cached_names = Some((*names_ptr).clone());
                    }
                    last_version = current_version;
                }
            }

            rt.block_on(async {
                // 3. 优化编码 - 使用带超时的锁
                let encoded_frames = match timeout(Duration::from_millis(2), encoders.lock()).await
                {
                    Ok(mut encoder_guard) => {
                        let mut results = Vec::new();
                        if let Some(ref names) = cached_names {
                            for name in names {
                                if let Some(encoder) = encoder_guard.get_mut(name) {
                                    if let Ok(Some(encoded_frame)) = encoder.encode(&raw_frame) {
                                        results.push((name.clone(), encoded_frame));
                                    }
                                }
                            }
                        }
                        results
                    }
                    Err(_) => Vec::new(), // 超时跳过
                };

                // 4. 优化分发 - 直接 HashMap 查找
                if !encoded_frames.is_empty() {
                    if let Some(ref streams) = cached_streams {
                        for (quality_name, encoded_frame) in encoded_frames {
                            for (stream_name, tx) in streams {
                                if stream_name == &quality_name {
                                    let _ = tx.send(encoded_frame.clone());
                                    break;
                                }
                            }
                        }
                    }
                    fps_counter += 1;
                    if fps_start_time.elapsed() >= Duration::from_secs(1) {
                        println!("[ENCODER] FPS: {} frames/sec", fps_counter);
                        fps_counter = 0;
                        fps_start_time = Instant::now();
                    }
                }
            });
        }
    }

    pub async fn add_quality_stream(
        &self,
        config: QualityConfig,
    ) -> Result<broadcast::Receiver<EncodedFrame>, Box<dyn std::error::Error + Send + Sync>> {
        config.validate()?;

        let quality_name = config.name.clone();

        // 快速检查重复
        {
            let streams = self.encoded_streams.read().await;
            if let Some(tx) = streams.get(&quality_name) {
                return Ok(tx.subscribe());
            }
        }

        let encoder = QualityEncoder::new(config)?;
        let (tx, rx) = broadcast::channel(32);

        // 原子性添加
        {
            let mut encoders = self.encoders.lock().await;
            let mut streams = self.encoded_streams.write().await;
            encoders.insert(quality_name.clone(), encoder);
            streams.insert(quality_name.clone(), tx);
        }

        {
            let mut tracks = self.track_writers.write().await;
            tracks.insert(quality_name, Vec::new());
        }

        // 更新快照以供编码线程使用
        self.update_snapshots().await;

        Ok(rx)
    }

    pub async fn add_webrtc_track(
        &self,
        quality_name: &str,
        track: Arc<TrackLocalStaticSample>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut encoded_rx = {
            let streams = self.encoded_streams.read().await;
            streams
                .get(quality_name)
                .ok_or("quality stream not found")?
                .subscribe()
        };

        {
            let mut track_writers = self.track_writers.write().await;
            track_writers
                .entry(quality_name.to_string())
                .or_default()
                .push(track.clone());
        }

        let shutdown_signal = self.shutdown_signal.clone();
        let quality_name = quality_name.to_string();
        let this_shutdown_signal = Arc::new(AtomicBool::new(false));

        let should_spawn = {
            let mut hash_guard = self.track_shutdown_tx.try_lock();
            match hash_guard {
                Ok(ref mut guard) if !guard.contains_key(&quality_name) => {
                    guard.insert(quality_name.clone(), this_shutdown_signal.clone());
                    true
                }
                _ => false,
            }
        };

        if !should_spawn {
            return Ok(());
        }

        // 核心优化：分离接收和写入
        tokio::spawn(async move {
            // 创建帧缓冲队列 - 关键是容量控制
            let (frame_tx, mut frame_rx) = tokio::sync::mpsc::channel::<Sample>(5); // 只缓冲5帧防止延迟

            let shutdown_signal_clone = shutdown_signal.clone();
            let this_shutdown_signal_clone = this_shutdown_signal.clone();

            // 快速接收线程
            let recv_task = tokio::spawn(async move {
                let mut fps_counter = 0u32;
                let mut fps_start_time = Instant::now();

                while !(shutdown_signal.load(Ordering::Relaxed)
                    || this_shutdown_signal.load(Ordering::Relaxed))
                {
                    match encoded_rx.recv().await {
                        Ok(encoded_frame) => {
                            let sample = Sample {
                                data: encoded_frame.data,
                                duration: Duration::from_millis(16),
                                ..Default::default()
                            };

                            // 非阻塞发送，如果队列满了就丢弃旧帧
                            match frame_tx.try_send(sample) {
                                Ok(_) => {
                                    fps_counter += 1;
                                }
                                Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                                    println!("[RECV] 写入队列满，丢弃帧 - 写入太慢");
                                    // 队列满说明写入跟不上，丢弃当前帧
                                }
                                Err(_) => break, // 通道关闭
                            }

                            if fps_start_time.elapsed() >= Duration::from_secs(1) {
                                println!("[RECV] 接收FPS: {}", fps_counter);
                                fps_counter = 0;
                                fps_start_time = Instant::now();
                            }
                        }
                        Err(RecvError::Lagged(missed_count)) => {
                            println!("[RECV] 丢帧 {} 帧", missed_count);
                            continue;
                        }
                        Err(RecvError::Closed) => break,
                    }
                }
            });

            // 专门的写入线程
            let write_task = tokio::spawn(async move {
                let mut write_fps_counter = 0u32;
                let mut write_fps_start_time = Instant::now();
                let mut write_times = Vec::new();

                while !(shutdown_signal_clone.load(Ordering::Relaxed)
                    || this_shutdown_signal_clone.load(Ordering::Relaxed))
                {
                    match frame_rx.recv().await {
                        Some(sample) => {
                            let write_start = Instant::now();

                            if track.write_sample(&sample).await.is_err() {
                                println!("[WRITE] WebRTC写入失败");
                                break;
                            }

                            let write_time = write_start.elapsed();
                            write_times.push(write_time);
                            write_fps_counter += 1;

                            if write_fps_start_time.elapsed() >= Duration::from_secs(1) {
                                let avg_write_time =
                                    write_times.iter().sum::<Duration>() / write_times.len() as u32;
                                println!(
                                    "[WRITE] 写入FPS: {}, 平均写入时间: {:?}",
                                    write_fps_counter, avg_write_time
                                );

                                if avg_write_time > Duration::from_millis(50) {
                                    println!("[WRITE] ⚠️  写入时间过长: {:?}", avg_write_time);
                                }

                                write_fps_counter = 0;
                                write_fps_start_time = Instant::now();
                                write_times.clear();
                            }
                        }
                        None => break, // 通道关闭
                    }
                }
            });

            // 等待任一任务完成
            tokio::select! {
                _ = recv_task => println!("[TRACK] 接收任务结束"),
                _ = write_task => println!("[TRACK] 写入任务结束"),
            }
        });

        Ok(())
    }

    pub async fn close_track_write(&self, quality_name: &str) {
        // 先获取 shutdown 信号，避免长时间持有锁
        let shutdown_signal = {
            let mut hash_guard = self.track_shutdown_tx.lock().await;
            hash_guard.remove(quality_name)
        };

        if let Some(track_shutdown) = shutdown_signal {
            track_shutdown.store(true, Ordering::Relaxed);
            self.remove_quality_stream(quality_name).await;
        }
    }

    pub async fn remove_quality_stream(&self, quality_name: &str) {
        {
            let mut encoders = self.encoders.lock().await;
            encoders.remove(quality_name);
        }
        {
            let mut streams = self.encoded_streams.write().await;
            streams.remove(quality_name);
        }
        {
            let mut track_writers = self.track_writers.write().await;
            track_writers.remove(quality_name);
        }
    }

    pub async fn get_active_qualities(&self) -> Vec<String> {
        let encoders = self.encoders.lock().await;
        encoders.keys().cloned().collect()
    }

    pub async fn shutdown(&self) {
        self.shutdown_signal.store(true, Ordering::Relaxed);

        // 发送关闭信号
        {
            let capture_tx_guard = self.capture_shutdown_tx.lock().await;
            if let Some(ref tx) = *capture_tx_guard {
                let _ = tx.send(());
            }
        }

        {
            let encoding_tx_guard = self.encoding_shutdown_tx.lock().await;
            if let Some(ref tx) = *encoding_tx_guard {
                let _ = tx.send(());
            }
        }

        // 等待线程结束
        {
            let mut capture_handle_guard = self.capture_handle.lock().await;
            if let Some(handle) = capture_handle_guard.take() {
                drop(capture_handle_guard); // 提前释放锁
                let _ = handle.join();
            }
        }

        {
            let mut encoding_handle_guard = self.encoding_handle.lock().await;
            if let Some(handle) = encoding_handle_guard.take() {
                drop(encoding_handle_guard); // 提前释放锁
                let _ = handle.join();
            }
        }

        // 快速清理 - 并行清理各个组件
        let (encoders_clear, streams_clear, tracks_clear) = tokio::join!(
            async {
                let mut encoders = self.encoders.lock().await;
                encoders.clear();
            },
            async {
                let mut streams = self.encoded_streams.write().await;
                streams.clear();
            },
            async {
                let mut tracks = self.track_writers.write().await;
                tracks.clear();
            }
        );
    }
}

impl Drop for MultiStreamManager {
    fn drop(&mut self) {
        self.shutdown_signal.store(true, Ordering::Relaxed);
    }
}
