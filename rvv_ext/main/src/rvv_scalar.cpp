#include "rvv.hpp"

#include <cmath>

namespace rvv::detail {

void add_f32_scalar(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

void sub_f32_scalar(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = a[i] - b[i];
}

void scale_f32_scalar(const float* a, float k, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = a[i] * k;
}

float dot_f32_scalar(const float* a, const float* b, size_t n) {
    double acc = 0.0;
    for (size_t i = 0; i < n; ++i) acc += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    return static_cast<float>(acc);
}

float norm_l2_f32_scalar(const float* a, size_t n) {
    return std::sqrt(dot_f32_scalar(a, a, n));
}

void matmul_f32_scalar(const float* A, const float* B, float* C, size_t m, size_t k, size_t n) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double acc = 0.0;
            for (size_t kk = 0; kk < k; ++kk) {
                acc += static_cast<double>(A[i * k + kk]) * static_cast<double>(B[kk * n + j]);
            }
            C[i * n + j] = static_cast<float>(acc);
        }
    }
}

} // namespace rvv::detail
