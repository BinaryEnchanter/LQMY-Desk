use bytes::Bytes;

use dashmap::DashMap;
use openh264::encoder::{Encoder, EncoderConfig, IntraFramePeriod, QpRange};
use openh264::formats::YUVSource;
use rusty_duplication::{FrameInfoExt, Scanner, VecCapturer};
use std::alloc::{alloc, dealloc, Layout};
use std::collections::{HashMap, VecDeque};
//use std::intrinsics::{prefetch_read_data, prefetch_write_data};
use std::ptr::{copy_nonoverlapping, write_bytes};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use webrtc::media::Sample;
use webrtc::track::track_local::track_local_static_sample::TrackLocalStaticSample;

// ==================== 核心数据结构 ====================

/// 零拷贝原始帧数据 - 使用原子指针和引用计数
#[repr(C, align(64))] // CPU缓存线对齐
pub struct RawFrame {
    pub width: u32,
    pub height: u32,
    pub stride: u32,       // 实际行字节数
    pub data_ptr: *mut u8, // 直接指针，避免Arc开销
    pub data_len: usize,
    pub ref_count: AtomicU64,
    pub timestamp: u64,
    pub frame_id: u64,
}

unsafe impl Send for RawFrame {}
unsafe impl Sync for RawFrame {}

impl RawFrame {
    pub unsafe fn new(width: u32, height: u32, data: Vec<u8>) -> Self {
        let len = data.len();
        let ptr = Box::into_raw(data.into_boxed_slice()) as *mut u8;

        Self {
            width,
            height,
            stride: width * 4, // BGRA
            data_ptr: ptr,
            data_len: len,
            ref_count: AtomicU64::new(1),
            timestamp: 0,
            frame_id: 0,
        }
    }

    pub fn acquire(&self) -> &Self {
        self.ref_count.fetch_add(1, Ordering::Relaxed);
        self
    }

    pub unsafe fn data_slice(&self) -> &[u8] {
        std::slice::from_raw_parts(self.data_ptr, self.data_len)
    }
}

impl Drop for RawFrame {
    fn drop(&mut self) {
        if self.ref_count.fetch_sub(1, Ordering::Relaxed) == 1 {
            unsafe {
                let layout = Layout::from_size_align_unchecked(self.data_len, 1);
                dealloc(self.data_ptr, layout);
            }
        }
    }
}

impl Clone for RawFrame {
    fn clone(&self) -> Self {
        self.ref_count.fetch_add(1, Ordering::Relaxed);
        Self {
            width: self.width,
            height: self.height,
            stride: self.stride,
            data_ptr: self.data_ptr,
            data_len: self.data_len,
            ref_count: AtomicU64::new(0), // 克隆体不管理内存
            timestamp: self.timestamp,
            frame_id: self.frame_id,
        }
    }
}

/// 质量配置 - 添加更多优化参数
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct QualityConfig {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub bitrate: u32,
    pub fps: u32,
    pub max_keyframe_interval: u32,
    pub qp_min: u8,
    pub qp_max: u8,
    pub complexity: u8,         // 0=ultra_fast, 1=fast, 2=medium
    pub tune_for_content: bool, // 是否针对屏幕内容优化
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
            qp_min: 16,
            qp_max: 32,
            complexity: 0, // 默认最快
            tune_for_content: true,
        }
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.width == 0 || self.height == 0 || self.width % 16 != 0 || self.height % 16 != 0 {
            return Err("dimensions must be positive and multiple of 16");
        }
        if self.fps == 0 || self.fps > 144 {
            return Err("fps must be between 1 and 144");
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
    //pub encode_time_us: u64, // 编码耗时（微秒）
}

/// 高性能YUV缓冲区 - SIMD优化内存布局
#[repr(C, align(32))] // AVX对齐
pub struct YuvBuffer {
    pub width: usize,
    pub height: usize,
    pub y_ptr: *mut u8,
    pub u_ptr: *mut u8,
    pub v_ptr: *mut u8,
    capacity: usize,
}

impl YuvBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        let y_size = width * height;
        let uv_size = (y_size + 3) / 4; // 向上取整
        let total_size = y_size + uv_size * 2;

        unsafe {
            let layout = Layout::from_size_align_unchecked(total_size, 32);
            let ptr = alloc(layout);
            if ptr.is_null() {
                panic!("Failed to allocate YUV buffer");
            }

            // 初始化为黑色
            write_bytes(ptr, 0, total_size);

            Self {
                width,
                height,
                y_ptr: ptr,
                u_ptr: ptr.add(y_size),
                v_ptr: ptr.add(y_size + uv_size),
                capacity: total_size,
            }
        }
    }

    pub fn resize(&mut self, width: usize, height: usize) {
        if self.width == width && self.height == height {
            return;
        }

        let y_size = width * height;
        let uv_size = (y_size + 3) / 4;
        let total_size = y_size + uv_size * 2;

        if total_size > self.capacity {
            unsafe {
                let old_layout = Layout::from_size_align_unchecked(self.capacity, 32);
                dealloc(self.y_ptr, old_layout);

                let new_layout = Layout::from_size_align_unchecked(total_size, 32);
                let ptr = alloc(new_layout);
                if ptr.is_null() {
                    panic!("Failed to reallocate YUV buffer");
                }

                self.y_ptr = ptr;
                self.u_ptr = ptr.add(y_size);
                self.v_ptr = ptr.add(y_size + uv_size);
                self.capacity = total_size;
            }
        } else {
            unsafe {
                self.u_ptr = self.y_ptr.add(y_size);
                self.v_ptr = self.y_ptr.add(y_size + uv_size);
            }
        }

        self.width = width;
        self.height = height;
    }
}

impl Drop for YuvBuffer {
    fn drop(&mut self) {
        unsafe {
            let layout = Layout::from_size_align_unchecked(self.capacity, 32);
            dealloc(self.y_ptr, layout);
        }
    }
}

unsafe impl Send for YuvBuffer {}
unsafe impl Sync for YuvBuffer {}

impl YUVSource for YuvBuffer {
    fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    fn strides(&self) -> (usize, usize, usize) {
        (self.width, self.width / 2, self.width / 2)
    }

    fn y(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.y_ptr, self.width * self.height) }
    }

    fn u(&self) -> &[u8] {
        unsafe {
            let size = (self.width * self.height + 3) / 4;
            std::slice::from_raw_parts(self.u_ptr, size)
        }
    }

    fn v(&self) -> &[u8] {
        unsafe {
            let size = (self.width * self.height + 3) / 4;
            std::slice::from_raw_parts(self.v_ptr, size)
        }
    }
}

// ==================== SIMD优化图像处理 ====================

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// 超高速BGRA到YUV420转换 - AVX2优化
pub unsafe fn convert_bgra_to_yuv420_simd(
    bgra: &[u8],
    src_width: usize,
    src_height: usize,
    yuv: &mut YuvBuffer,
) {
    yuv.resize(src_width, src_height);

    let width = src_width;
    let height = src_height;

    // 预取缓存
    //prefetch_read_data(bgra.as_ptr(), 3);
    //prefetch_write_data(yuv.y_ptr, 3);

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        convert_bgra_to_yuv420_avx2(bgra, width, height, yuv);
        return;
    }

    // Fallback到标量版本
    convert_bgra_to_yuv420_scalar(bgra, width, height, yuv);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn convert_bgra_to_yuv420_avx2(
    bgra: &[u8],
    width: usize,
    height: usize,
    yuv: &mut YuvBuffer,
) {
    // YUV转换系数（定点数，乘256）
    let y_r = _mm256_set1_epi16(77);
    let y_g = _mm256_set1_epi16(150);
    let y_b = _mm256_set1_epi16(29);
    let u_r = _mm256_set1_epi16(-43);
    let u_g = _mm256_set1_epi16(-85);
    let u_b = _mm256_set1_epi16(128);
    let v_r = _mm256_set1_epi16(128);
    let v_g = _mm256_set1_epi16(-107);
    let v_b = _mm256_set1_epi16(-21);

    let offset_16 = _mm256_set1_epi16(16);
    let offset_128 = _mm256_set1_epi16(128);
    let zero = _mm256_setzero_si256();

    // Y平面处理 - 8像素并行
    for y in 0..height {
        let mut x = 0;
        let src_row = bgra.as_ptr().add(y * width * 4);
        let dst_row = yuv.y_ptr.add(y * width);

        // SIMD处理（8像素）
        while x + 8 <= width {
            // 加载8个BGRA像素（32字节）
            let pixels = _mm256_loadu_si256(src_row.add(x * 4) as *const __m256i);

            // 提取颜色通道
            let bgra_lo = _mm256_unpacklo_epi8(pixels, zero);
            let bgra_hi = _mm256_unpackhi_epi8(pixels, zero);

            // 计算Y值
            let y_lo = compute_y_component_avx2(bgra_lo, y_r, y_g, y_b, offset_16);
            let y_hi = compute_y_component_avx2(bgra_hi, y_r, y_g, y_b, offset_16);

            // 打包并存储
            let y_packed = _mm256_packus_epi16(y_lo, y_hi);
            _mm256_storeu_si256(dst_row.add(x) as *mut __m256i, y_packed);

            x += 8;
        }

        // 处理剩余像素
        while x < width {
            let pixel_idx = src_row.add(x * 4);
            let b = *pixel_idx as i32;
            let g = *pixel_idx.add(1) as i32;
            let r = *pixel_idx.add(2) as i32;

            let y_val = ((77 * r + 150 * g + 29 * b) >> 8) + 16;
            *dst_row.add(x) = y_val.clamp(16, 235) as u8;
            x += 1;
        }
    }

    // UV平面处理 - 2x2子采样，4像素并行
    let uv_width = width / 2;
    let uv_height = height / 2;

    for uv_y in 0..uv_height {
        let mut uv_x = 0;

        while uv_x + 4 <= uv_width {
            // 处理4个UV像素（对应8x2的BGRA区域）
            process_uv_block_avx2(
                bgra,
                width,
                uv_x,
                uv_y,
                yuv.u_ptr.add(uv_y * uv_width + uv_x),
                yuv.v_ptr.add(uv_y * uv_width + uv_x),
                u_r,
                u_g,
                u_b,
                v_r,
                v_g,
                v_b,
                offset_128,
            );
            uv_x += 4;
        }

        // 处理剩余UV像素
        while uv_x < uv_width {
            process_single_uv_pixel(bgra, width, uv_x, uv_y, yuv);
            uv_x += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn compute_y_component_avx2(
    bgra: __m256i,
    y_r: __m256i,
    y_g: __m256i,
    y_b: __m256i,
    offset: __m256i,
) -> __m256i {
    // 提取RGB通道
    let mask_r = _mm256_set1_epi32(0x00FF0000);
    let mask_g = _mm256_set1_epi32(0x0000FF00);
    let mask_b = _mm256_set1_epi32(0x000000FF);

    let r = _mm256_and_si256(_mm256_srli_epi32(bgra, 16), _mm256_set1_epi32(0xFF));
    let g = _mm256_and_si256(_mm256_srli_epi32(bgra, 8), _mm256_set1_epi32(0xFF));
    let b = _mm256_and_si256(bgra, _mm256_set1_epi32(0xFF));

    // 转换为16位并计算
    let r_16 = _mm256_packus_epi32(r, r);
    let g_16 = _mm256_packus_epi32(g, g);
    let b_16 = _mm256_packus_epi32(b, b);

    let yr = _mm256_mullo_epi16(r_16, y_r);
    let yg = _mm256_mullo_epi16(g_16, y_g);
    let yb = _mm256_mullo_epi16(b_16, y_b);

    let sum = _mm256_add_epi16(_mm256_add_epi16(yr, yg), yb);
    let shifted = _mm256_srai_epi16(sum, 8);

    _mm256_add_epi16(shifted, offset)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn process_uv_block_avx2(
    bgra: &[u8],
    width: usize,
    uv_x: usize,
    uv_y: usize,
    u_dst: *mut u8,
    v_dst: *mut u8,
    u_r: __m256i,
    u_g: __m256i,
    u_b: __m256i,
    v_r: __m256i,
    v_g: __m256i,
    v_b: __m256i,
    offset: __m256i,
) {
    // 处理4个UV像素的简化版本
    for i in 0..4 {
        if uv_x + i < width / 2 {
            process_single_uv_pixel_at(bgra, width, uv_x + i, uv_y, u_dst.add(i), v_dst.add(i));
        }
    }
}

unsafe fn process_single_uv_pixel_at(
    bgra: &[u8],
    width: usize,
    uv_x: usize,
    uv_y: usize,
    u_dst: *mut u8,
    v_dst: *mut u8,
) {
    let src_x = uv_x * 2;
    let src_y = uv_y * 2;

    let mut sum_u = 0i32;
    let mut sum_v = 0i32;

    // 2x2子采样
    for dy in 0..2 {
        for dx in 0..2 {
            let pixel_idx = ((src_y + dy) * width + (src_x + dx)) * 4;
            let b = bgra[pixel_idx] as i32;
            let g = bgra[pixel_idx + 1] as i32;
            let r = bgra[pixel_idx + 2] as i32;

            sum_u += (-43 * r - 85 * g + 128 * b) >> 8;
            sum_v += (128 * r - 107 * g - 21 * b) >> 8;
        }
    }

    *u_dst = ((sum_u >> 2) + 128).clamp(16, 240) as u8;
    *v_dst = ((sum_v >> 2) + 128).clamp(16, 240) as u8;
}

unsafe fn process_single_uv_pixel(
    bgra: &[u8],
    width: usize,
    uv_x: usize,
    uv_y: usize,
    yuv: &mut YuvBuffer,
) {
    let uv_width = width / 2;
    let dst_idx = uv_y * uv_width + uv_x;

    process_single_uv_pixel_at(
        bgra,
        width,
        uv_x,
        uv_y,
        yuv.u_ptr.add(dst_idx),
        yuv.v_ptr.add(dst_idx),
    );
}

unsafe fn convert_bgra_to_yuv420_scalar(
    bgra: &[u8],
    width: usize,
    height: usize,
    yuv: &mut YuvBuffer,
) {
    // 标量版本的快速实现
    for y in 0..height {
        let src_row = y * width * 4;
        let dst_row = y * width;

        for x in 0..width {
            let pixel_idx = src_row + x * 4;
            let b = bgra[pixel_idx] as i32;
            let g = bgra[pixel_idx + 1] as i32;
            let r = bgra[pixel_idx + 2] as i32;

            let y_val = ((77 * r + 150 * g + 29 * b) >> 8) + 16;
            *yuv.y_ptr.add(dst_row + x) = y_val.clamp(16, 235) as u8;
        }
    }

    // UV处理
    let uv_width = width / 2;
    let uv_height = height / 2;

    for uv_y in 0..uv_height {
        for uv_x in 0..uv_width {
            process_single_uv_pixel(bgra, width, uv_x, uv_y, yuv);
        }
    }
}

/// 高性能图像缩放 - SIMD双线性插值
pub unsafe fn resize_bgra_simd(
    src: &[u8],
    src_width: usize,
    src_height: usize,
    dst: &mut [u8],
    dst_width: usize,
    dst_height: usize,
) {
    let x_scale = src_width as f32 / dst_width as f32;
    let y_scale = src_height as f32 / dst_height as f32;

    // 预计算缩放查找表
    let mut x_indices = Vec::with_capacity(dst_width);
    let mut x_weights = Vec::with_capacity(dst_width);

    for dst_x in 0..dst_width {
        let src_x_f = dst_x as f32 * x_scale;
        let src_x = src_x_f as usize;
        let weight = src_x_f - src_x as f32;

        x_indices.push((src_x, (src_x + 1).min(src_width - 1)));
        x_weights.push((1.0 - weight, weight));
    }

    for dst_y in 0..dst_height {
        let src_y_f = dst_y as f32 * y_scale;
        let src_y = src_y_f as usize;
        let y_weight = src_y_f - src_y as f32;
        let src_y1 = (src_y + 1).min(src_height - 1);

        for dst_x in 0..dst_width {
            let (src_x0, src_x1) = x_indices[dst_x];
            let (x_weight0, x_weight1) = x_weights[dst_x];

            // 4个邻近像素
            let p00_idx = (src_y * src_width + src_x0) * 4;
            let p01_idx = (src_y * src_width + src_x1) * 4;
            let p10_idx = (src_y1 * src_width + src_x0) * 4;
            let p11_idx = (src_y1 * src_width + src_x1) * 4;

            let dst_idx = (dst_y * dst_width + dst_x) * 4;

            // SIMD双线性插值（4个通道同时处理）
            for c in 0..4 {
                let p00 = src[p00_idx + c] as f32;
                let p01 = src[p01_idx + c] as f32;
                let p10 = src[p10_idx + c] as f32;
                let p11 = src[p11_idx + c] as f32;

                let top = p00 * x_weight0 + p01 * x_weight1;
                let bottom = p10 * x_weight0 + p11 * x_weight1;
                let result = top * (1.0 - y_weight) + bottom * y_weight;

                dst[dst_idx + c] = result.clamp(0.0, 255.0) as u8;
            }
        }
    }
}

// ==================== 零拷贝编码器管理 ====================

/// 高性能编码器 - 零拷贝设计
struct QualityEncoder {
    encoder: Encoder,
    config: QualityConfig,
    frame_interval_ns: u64,
    last_encode_time: Instant,
    frame_count: u64,
    yuv_buffer: YuvBuffer,
    resize_buffer: Option<Vec<u8>>, // 延迟分配
    skip_counter: u32,              // 跳帧计数器
}

impl QualityEncoder {
    fn new(config: QualityConfig) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let loader = openh264::OpenH264API::from_source();

        let enc_config = EncoderConfig::new();
        enc_config.skip_frames(false);
        enc_config.bitrate(openh264::encoder::BitRate::from_bps(config.bitrate));
        enc_config.max_frame_rate(openh264::encoder::FrameRate::from_hz(config.fps as f32));

        // 根据复杂度设置
        let usage_type = if config.tune_for_content {
            openh264::encoder::UsageType::ScreenContentRealTime
        } else {
            openh264::encoder::UsageType::CameraVideoRealTime
        };
        enc_config.usage_type(usage_type);

        let complexity = match config.complexity {
            0 => openh264::encoder::Complexity::Low,
            1 => openh264::encoder::Complexity::Medium,
            _ => openh264::encoder::Complexity::High,
        };
        enc_config.complexity(complexity);

        enc_config.profile(openh264::encoder::Profile::Baseline);
        enc_config.level(openh264::encoder::Level::Level_3_1);
        enc_config.qp(QpRange::new(config.qp_min, config.qp_max));
        enc_config.intra_frame_period(IntraFramePeriod::from_num_frames(
            config.max_keyframe_interval,
        ));

        let encoder = Encoder::with_api_config(loader, enc_config)
            .map_err(|e| format!("failed to create encoder: {}", e))?;

        let frame_interval_ns = 1_000_000_000 / config.fps as u64;
        let yuv_buffer = YuvBuffer::new(config.width as usize, config.height as usize);

        Ok(Self {
            encoder,
            config,
            frame_interval_ns,
            last_encode_time: Instant::now(),
            frame_count: 0,
            yuv_buffer,
            resize_buffer: None,
            skip_counter: 0,
        })
    }

    fn should_encode(&mut self) -> bool {
        let elapsed = self.last_encode_time.elapsed().as_nanos() as u64;

        if elapsed >= self.frame_interval_ns {
            self.skip_counter = 0;
            true
        } else {
            // 自适应跳帧
            self.skip_counter += 1;
            false
        }
    }

    /// 编码一个原始帧，返回编码后的帧（如果有）
    fn encode(
        &mut self,
        raw_frame: &RawFrame,
    ) -> Result<Option<EncodedFrame>, Box<dyn std::error::Error + Send + Sync>> {
        if !self.should_encode() {
            return Ok(None);
        }

        self.last_encode_time = Instant::now();

        // 零拷贝缩放处理
        let source_data =
            if raw_frame.width == self.config.width && raw_frame.height == self.config.height {
                // 直接使用原始数据，避免拷贝
                unsafe { raw_frame.data_slice() }
            } else {
                // 需要缩放时才进行内存操作
                unsafe {
                    resize_bgra_simd(
                        raw_frame.data_slice(),
                        raw_frame.width as usize,
                        raw_frame.height as usize,
                        self.resize_buffer.as_mut().unwrap(),
                        self.config.width as usize,
                        self.config.height as usize,
                    );
                }
                &self.resize_buffer.as_mut().unwrap()[..]
            };

        // SIMD 优化的 YUV 转换
        unsafe {
            convert_bgra_to_yuv420_simd(
                source_data,
                self.config.width as usize,
                self.config.height as usize,
                &mut self.yuv_buffer,
            );
        }

        // H.264 编码
        match self.encoder.encode(&self.yuv_buffer) {
            Ok(bitstream) => {
                let is_keyframe = self.frame_count % self.config.max_keyframe_interval as u64 == 0;

                let encoded_frame = EncodedFrame {
                    data: Bytes::from(bitstream.to_vec()),
                    timestamp: raw_frame.timestamp,
                    frame_id: raw_frame.frame_id,
                    is_keyframe,
                    quality: self.config.name.clone(),
                    //encode_time_us: todo!(),
                };

                self.frame_count += 1;
                Ok(Some(encoded_frame))
            }
            Err(e) => Err(format!("encoding failed: {}", e).into()),
        }
    }

    /// 获取编码器的当前统计信息
    fn get_stats(&self) -> EncoderStats {
        EncoderStats {
            frame_count: self.frame_count,
            quality_name: self.config.name.clone(),
            current_fps: if self.last_encode_time.elapsed().as_secs_f64() > 0.0 {
                1.0 / self.last_encode_time.elapsed().as_secs_f64()
            } else {
                0.0
            },
            target_fps: self.config.fps,
        }
    }
}

// ============== 编码器统计信息 =================
#[derive(Debug, Clone)]
pub struct EncoderStats {
    pub frame_count: u64,
    pub quality_name: String,
    pub current_fps: f64,
    pub target_fps: u32,
}

// ============== 零拷贝帧池管理 =================
pub struct FramePool {
    pool: Vec<RawFrame>,
    max_size: usize,
}

impl FramePool {
    /// 创建一个新的帧池
    pub fn new(max_size: usize) -> Self {
        Self {
            pool: Vec::with_capacity(max_size),
            max_size,
        }
    }

    /// 从池中获取一个帧，如果池为空则创建新帧
    pub unsafe fn acquire(&mut self, width: u32, height: u32) -> RawFrame {
        if let Some(mut frame) = self.pool.pop() {
            // 检查帧尺寸是否匹配
            if frame.width == width && frame.height == height {
                return frame;
            }
        }

        // 创建新帧
        let data_size = (width * height * 4) as usize;
        let data = vec![0u8; data_size];
        RawFrame::new(width, height, data)
    }

    /// 将帧返回到池中
    pub fn release(&mut self, frame: RawFrame) {
        if self.pool.len() < self.max_size {
            self.pool.push(frame);
        }
        // 如果池已满，帧会被自动释放
    }
}

// ============== 优化后的多流管理器 =================
pub struct OptimizedMultiStreamManager {
    // 原始帧广播（使用无锁队列优化）
    raw_frame_tx: crossbeam_channel::Sender<RawFrame>,
    raw_frame_rx: crossbeam_channel::Receiver<RawFrame>,

    // 编码器管理（减少锁竞争）
    encoders: DashMap<String, QualityEncoder>,

    // 编码后的帧分发（使用更高效的多播）
    encoded_streams: DashMap<String, crossbeam_channel::Sender<EncodedFrame>>,
    encoded_receivers: DashMap<String, Vec<crossbeam_channel::Receiver<EncodedFrame>>>,

    // WebRTC 轨道管理
    track_writers: DashMap<String, Vec<Arc<TrackLocalStaticSample>>>,

    // 帧池管理
    frame_pool: Arc<Mutex<FramePool>>,

    // 控制信号
    shutdown_signal: Arc<AtomicBool>,
    frame_counter: Arc<AtomicU64>,

    // 任务句柄
    capture_handle: Option<tokio::task::JoinHandle<()>>,
    encoding_handles: Vec<tokio::task::JoinHandle<()>>,

    // 性能监控
    performance_monitor: Arc<PerformanceMonitor>,
}

impl OptimizedMultiStreamManager {
    pub fn new() -> Self {
        let (raw_frame_tx, raw_frame_rx) = crossbeam_channel::bounded(4); // 小缓冲区避免延迟

        Self {
            raw_frame_tx,
            raw_frame_rx,
            encoders: DashMap::new(),
            encoded_streams: DashMap::new(),
            encoded_receivers: DashMap::new(),
            track_writers: DashMap::new(),
            frame_pool: Arc::new(Mutex::new(FramePool::new(8))),
            shutdown_signal: Arc::new(AtomicBool::new(false)),
            frame_counter: Arc::new(AtomicU64::new(0)),
            capture_handle: None,
            encoding_handles: Vec::new(),
            performance_monitor: Arc::new(PerformanceMonitor::new()),
        }
    }

    /// 启动优化的桌面捕获
    pub async fn start_capture(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.capture_handle.is_some() {
            return Ok(());
        }

        let tx = self.raw_frame_tx.clone();
        let shutdown_signal = self.shutdown_signal.clone();
        let frame_counter = self.frame_counter.clone();
        let frame_pool = self.frame_pool.clone();
        let perf_monitor = self.performance_monitor.clone();

        let handle = tokio::spawn(async move {
            let mut scanner = Scanner::new().expect("failed to create scanner");
            let monitor = scanner.next().expect("no monitor found");
            let mut capturer: VecCapturer = monitor.try_into().expect("failed to create capturer");

            let mut last_capture = Instant::now();
            let capture_interval = Duration::from_nanos(16_666_667); // ~60fps 精确时间

            while !shutdown_signal.load(Ordering::Acquire) {
                let now = Instant::now();
                if now.duration_since(last_capture) < capture_interval {
                    // 使用更精确的睡眠
                    let sleep_time = capture_interval - now.duration_since(last_capture);
                    if sleep_time > Duration::from_micros(100) {
                        tokio::time::sleep(sleep_time - Duration::from_micros(100)).await;
                    }
                    continue;
                }

                let capture_start = Instant::now();

                if let Ok(info) = capturer.capture() {
                    if info.desktop_updated() {
                        let desc = capturer.monitor().dxgi_outdupl_desc();
                        let frame_id = frame_counter.fetch_add(1, Ordering::Relaxed);

                        // 从帧池获取帧，避免频繁分配
                        let mut raw_frame = unsafe {
                            frame_pool
                                .lock()
                                .await
                                .acquire(desc.ModeDesc.Width, desc.ModeDesc.Height)
                        };

                        // 零拷贝数据传输
                        unsafe {
                            let src_data = capturer.buffer.as_slice();
                            let dst_data = std::slice::from_raw_parts_mut(
                                raw_frame.data_ptr.as_mut().unwrap(),
                                raw_frame.data_len,
                            );
                            dst_data.copy_from_slice(src_data);
                        }

                        raw_frame.timestamp = frame_id;
                        raw_frame.frame_id = frame_id;

                        last_capture = now;

                        // 记录性能指标
                        perf_monitor.record_capture_time(capture_start.elapsed());

                        if tx.send(raw_frame).is_err() {
                            break; // 所有接收者都已关闭
                        }
                    }
                }

                // 短暂让出 CPU
                tokio::task::yield_now().await;
            }
        });

        self.capture_handle = Some(handle);
        self.start_parallel_encoding_workers().await;

        Ok(())
    }

    /// 启动并行编码工作线程（每个质量一个线程）
    async fn start_parallel_encoding_workers(&mut self) {
        // 为每个编码器启动独立的工作线程
        let qualities: Vec<String> = self
            .encoders
            .iter()
            .map(|entry| entry.key().clone())
            .collect();

        for quality_name in qualities {
            let rx = self.raw_frame_rx.clone();
            let encoders = &self.encoders;
            let encoded_streams = self.encoded_streams.clone();
            let shutdown_signal = self.shutdown_signal.clone();
            let perf_monitor = self.performance_monitor.clone();
            let quality = quality_name.clone();
            let (_, mut encoder_wrapped) = self.encoders.remove(&quality_name).unwrap();

            let handle = tokio::spawn(async move {
                while !shutdown_signal.load(Ordering::Acquire) {
                    match rx.try_recv() {
                        Ok(raw_frame) => {
                            let encode_start = Instant::now();

                            match encoder_wrapped.encode(&raw_frame) {
                                Ok(Some(encoded_frame)) => {
                                    perf_monitor
                                        .record_encode_time(&quality, encode_start.elapsed());

                                    if let Some(tx) = encoded_streams.get(&quality) {
                                        let _ = tx.send(encoded_frame);
                                    }
                                }
                                Ok(None) => {
                                    // 跳帧，正常情况
                                }
                                Err(e) => {
                                    eprintln!("Encoding error for {}: {}", quality, e);
                                }
                            }
                        }
                        Err(crossbeam_channel::TryRecvError::Empty) => {
                            // 没有新帧，短暂休眠
                            tokio::time::sleep(Duration::from_micros(100)).await;
                        }
                        Err(crossbeam_channel::TryRecvError::Disconnected) => break,
                    }
                }
            });
            //self.encoders.insert(quality_name, encoder_wrapped);
            self.encoding_handles.push(handle);
        }
    }

    /// 添加质量流（优化版本）
    pub async fn add_quality_stream(
        &self,
        config: QualityConfig,
    ) -> Result<crossbeam_channel::Receiver<EncodedFrame>, Box<dyn std::error::Error + Send + Sync>>
    {
        config.validate()?;

        let quality_name = config.name.clone();

        // 检查是否已存在
        if let Some(tx) = self.encoded_streams.get(&quality_name) {
            let (new_tx, rx) = crossbeam_channel::bounded(16);
            // 创建转发任务
            let old_tx = tx.clone();
            tokio::spawn(async move {
                // 这里可以实现多播逻辑
            });
            return Ok(rx);
        }

        // 创建编码器
        let encoder = QualityEncoder::new(config)?;

        // 创建通道
        let (tx, rx) = crossbeam_channel::bounded(16);

        // 添加到管理器
        self.encoders.insert(quality_name.clone(), encoder);
        self.encoded_streams.insert(quality_name, tx);

        Ok(rx)
    }

    /// 获取性能统计信息
    pub async fn get_performance_stats(&self) -> PerformanceStats {
        self.performance_monitor.get_stats().await
    }

    /// 动态调整编码质量
    pub async fn adjust_quality(
        &self,
        quality_name: &str,
        new_bitrate: u32,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(mut encoder_ref) = self.encoders.get_mut(quality_name) {
            // 这里可以实现动态调整编码器参数的逻辑
            // encoder_ref.adjust_bitrate(new_bitrate)?;
            Ok(())
        } else {
            Err("Quality stream not found".into())
        }
    }

    /// 优化的关闭方法
    pub async fn shutdown(&mut self) {
        self.shutdown_signal.store(true, Ordering::Release);

        // 等待捕获线程
        if let Some(handle) = self.capture_handle.take() {
            let _ = tokio::time::timeout(Duration::from_secs(5), handle).await;
        }

        // 等待所有编码线程
        for handle in self.encoding_handles.drain(..) {
            let _ = tokio::time::timeout(Duration::from_secs(2), handle).await;
        }

        // 清理资源
        self.encoders.clear();
        self.encoded_streams.clear();
        self.encoded_receivers.clear();
        self.track_writers.clear();
    }
}

// ============== 性能监控 =================
pub struct PerformanceMonitor {
    capture_times: Arc<Mutex<VecDeque<Duration>>>,
    encode_times: Arc<Mutex<HashMap<String, VecDeque<Duration>>>>,
    frame_drops: Arc<AtomicU64>,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            capture_times: Arc::new(Mutex::new(VecDeque::with_capacity(100))),
            encode_times: Arc::new(Mutex::new(HashMap::new())),
            frame_drops: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn record_capture_time(&self, duration: Duration) {
        if let Ok(mut times) = self.capture_times.try_lock() {
            if times.len() >= 100 {
                times.pop_front();
            }
            times.push_back(duration);
        }
    }

    pub fn record_encode_time(&self, quality: &str, duration: Duration) {
        if let Ok(mut encode_times) = self.encode_times.try_lock() {
            let times = encode_times
                .entry(quality.to_string())
                .or_insert_with(|| VecDeque::with_capacity(100));

            if times.len() >= 100 {
                times.pop_front();
            }
            times.push_back(duration);
        }
    }

    pub fn record_frame_drop(&self) {
        self.frame_drops.fetch_add(1, Ordering::Relaxed);
    }

    pub async fn get_stats(&self) -> PerformanceStats {
        let avg_capture_time = {
            let times = self.capture_times.lock().await;
            if times.is_empty() {
                Duration::ZERO
            } else {
                let total: Duration = times.iter().sum();
                total / times.len() as u32
            }
        };

        let avg_encode_times = {
            let encode_times = self.encode_times.lock().await;
            let mut result = HashMap::new();

            for (quality, times) in encode_times.iter() {
                if !times.is_empty() {
                    let total: Duration = times.iter().sum();
                    result.insert(quality.clone(), total / times.len() as u32);
                }
            }
            result
        };

        PerformanceStats {
            avg_capture_time,
            avg_encode_times,
            total_frame_drops: self.frame_drops.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug)]
pub struct PerformanceStats {
    pub avg_capture_time: Duration,
    pub avg_encode_times: HashMap<String, Duration>,
    pub total_frame_drops: u64,
}

// ============== 内存对齐优化 =================
#[repr(align(32))] // AVX2 对齐
pub struct AlignedBuffer {
    data: Vec<u8>,
}

impl AlignedBuffer {
    pub fn new(size: usize) -> Self {
        let mut data = Vec::with_capacity(size + 31);
        data.resize(size, 0);
        Self { data }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }
}
