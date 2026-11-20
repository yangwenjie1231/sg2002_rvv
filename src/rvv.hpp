#pragma once
#include <cstddef>
#include <cstdint>
#include <utility>

namespace rvv::core {

/**
 * 向量加法 c = a + b
 * @param a   输入向量 a
 * @param b   输入向量 b
 * @param c   输出向量
 * @param n   元素个数
 * @param module rvv.core.add
 */
void add(const float* a, const float* b, float* c, std::size_t n);

/**
 * 向量减法 c = a - b
 * @module rvv.core.sub
 */
void sub(const float* a, const float* b, float* c, std::size_t n);

/**
 * 标量乘法 b = k * a
 * @module rvv.core.scale
 */
void scale(const float* a, float k, float* b, std::size_t n);

/**
 * 向量点积
 * @return 点积结果
 * @module rvv.core.dot
 */
float dot(const float* a, const float* b, std::size_t n);

/**
 * L2 范数 ||a||_2
 * @module rvv.core.norm_l2
 */
float norm_l2(const float* a, std::size_t n);

/**
 * 向量归一化 b = a / ||a||_2
 * @module rvv.core.normalize
 */
void normalize(const float* a, float* b, std::size_t n);

/**
 * 矩阵加法 C = A + B
 * @param rows 行数
 * @param cols 列数
 * @module rvv.core.add2d
 */
void add2d(const float* A, const float* B, float* C,
           std::size_t rows, std::size_t cols);

/**
 * 矩阵标量乘法 B = k * A
 * @module rvv.core.scale2d
 */
void scale2d(const float* A, float k, float* B,
             std::size_t rows, std::size_t cols);

/**
 * 矩阵乘法 C = A * B
 * A:[rows×k]  B:[k×cols]  → C:[rows×cols]
 * @module rvv.core.matmul
 */
void matmul(const float* A, const float* B, float* C,
            std::size_t rows, std::size_t k, std::size_t cols);

/**
 * 矩阵转置 B = A^T
 * @module rvv.core.transpose
 */
void transpose(const float* A, float* B,
               std::size_t rows, std::size_t cols);

/**
 * 矩阵 × 向量  y = A * x
 * A:[rows×cols]  x:[cols]  → y:[rows]
 * @module rvv.core.mv
 */
void mv(const float* A, const float* x, float* y,
        std::size_t rows, std::size_t cols);

// ------------------------------------------------------------------
// int8 向量/矩阵运算（新增）
// ------------------------------------------------------------------
/**
 * int8 向量加法 c = a + b
 * @module rvv.core.add_i8
 */
void add_i8(const int8_t* a, const int8_t* b, int8_t* c, std::size_t n);

/**
 * int8 标量乘法 b = k * a
 * @module rvv.core.scale_i8
 */
void scale_i8(const int8_t* a, int8_t k, int8_t* b, std::size_t n);

/**
 * int8 向量点积（累加到 int32）
 * @module rvv.core.dot_i8
 */
int32_t dot_i8(const int8_t* a, const int8_t* b, std::size_t n);

/**
 * int8 矩阵加法 C = A + B
 * @module rvv.core.add2d_i8
 */
void add2d_i8(const int8_t* A, const int8_t* B, int8_t* C,
              std::size_t rows, std::size_t cols);

/**
 * int8 矩阵标量乘法 B = k * A
 * @module rvv.core.scale2d_i8
 */
void scale2d_i8(const int8_t* A, int8_t k, int8_t* B,
                std::size_t rows, std::size_t cols);

// ------------------------------------------------------------------
// uint8 图像统计（对焦诊断）
// ------------------------------------------------------------------
/**
 * 计算对焦清晰度所需的原始统计量（不直接返回 score）。
 *
 * 统计口径与固件现有 Python 实现
 * `firmware/utils/image_utils.py::evaluate_sharpness_diagnostics` 对齐：
 *
 *   roi_w = max(4, int(width  * roi_fraction))
 *   roi_h = max(4, int(height * roi_fraction))
 *   x0    = (width  - roi_w) / 2
 *   y0    = (height - roi_h) / 2
 *   step_x = step_x == 0 ? max(2, roi_w / 100) : step_x
 *   step_y = step_y == 0 ? max(2, roi_h / 100) : step_y
 *
 * 对 ROI 内每个采样点 (x, y)：
 *   c  = 灰度(x,   y  )
 *   cx = 灰度(x+offset, y  )
 *   cy = 灰度(x,   y+offset)
 *   grad_sum    += (cx-c)^2 + (cy-c)^2
 *   lum_sum     += c
 *   lum_sq_sum  += c*c
 *   count       += 1
 *   saturated_count += (c >= 250)
 *
 * `channels == 1` 时输入为 8-bit 灰度；`channels == 3` 时输入为
 * RGB888，并按 `(R + G + B) / 3.0` 转灰度（不是 Rec.601）。
 * 越界的 `x+offset` / `y+offset` 采样点会被跳过；`width < 4`、
 * `height < 4` 或没有有效采样点时返回全零。
 *
 * Python 绑定返回 dict；其中 grad_sum/lum_sum/lum_sq_sum 为
 * Python float，count/saturated_count 为 Python int。
 *
 * @param data      uint8 数据首地址，按 data[y * stride + x * channels] 访问
 * @param width     图像宽（像素）
 * @param height    图像高（像素）
 * @param stride    每行字节数，必须 >= width * channels
 * @param roi_fraction 中心 ROI 比例，按 Python 实现钳制到 [0.1, 1.0]
 * @param step_x    水平采样步长；0 表示按 ROI 宽度自动计算
 * @param step_y    垂直采样步长；0 表示按 ROI 高度自动计算
 * @param offset    前向差分像素偏移（>= 1）
 * @param channels  1（灰度）或 3（RGB888）
 *
 * @note 本函数不分配图像大小的临时缓冲。RVV 实现仅使用两行
 *       float 灰度缓冲（2 * roi_w * sizeof(float)）。
 * @module rvv.core.gradient_stats_u8
 */
struct GradientStats {
    double grad_sum = 0.0;
    double lum_sum = 0.0;
    double lum_sq_sum = 0.0;
    std::uint64_t count = 0;
    std::uint64_t saturated_count = 0;
};

GradientStats gradient_stats_u8(const std::uint8_t* data,
                                std::size_t width,
                                std::size_t height,
                                std::size_t stride,
                                double roi_fraction,
                                std::size_t step_x,
                                std::size_t step_y,
                                std::size_t offset,
                                std::size_t channels);

// ------------------------------------------------------------------
// 通用 float32 扩展算子
// ------------------------------------------------------------------
/**
 * 逐元素绝对值 b = |a|
 * @module rvv.core.abs_f32
 */
void abs_f32(const float* a, float* b, std::size_t n);

/**
 * 求和 sum(a)，使用 double 累加
 * @return double 累加结果；空向量返回 0.0
 * @module rvv.core.sum_f32
 */
double sum_f32(const float* a, std::size_t n);

/**
 * 均值与总体标准差（population std，除以 n）
 * @param a 输入向量
 * @param n 元素个数
 * @return {mean, stddev}；n == 0 时返回 {NaN, NaN}
 * @module rvv.core.mean_std_f32
 */
std::pair<double, double> mean_std_f32(const float* a, std::size_t n);

/**
 * 逐元素乘法 c = a * b
 * @module rvv.core.mul_f32
 */
void mul_f32(const float* a, const float* b, float* c, std::size_t n);

/**
 * 逐元素加标量 b = a + k
 * @module rvv.core.add_scalar_f32
 */
void add_scalar_f32(const float* a, float k, float* b, std::size_t n);

/**
 * 逐元素截断 b = clamp(a, lo, hi)，保留 NaN
 * @param lo 下界
 * @param hi 上界（调用方需保证 lo <= hi；Python 绑定会校验）
 * @module rvv.core.clip_f32
 */
void clip_f32(const float* a, float lo, float hi, float* b, std::size_t n);

/**
 * 最大值；空向量返回 NaN，任一输入为 NaN 时返回 NaN
 * @module rvv.core.max_f32
 */
float max_f32(const float* a, std::size_t n);

/**
 * 最小值；空向量返回 NaN，任一输入为 NaN 时返回 NaN
 * @module rvv.core.min_f32
 */
float min_f32(const float* a, std::size_t n);

}  // namespace rvv::core