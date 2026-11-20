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
