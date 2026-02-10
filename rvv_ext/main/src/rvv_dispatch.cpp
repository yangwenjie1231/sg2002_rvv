#include "rvv.hpp"

namespace rvv {

void add_f32(const float* a, const float* b, float* out, size_t n) {
#if defined(__riscv_vector)
    detail::add_f32_rvv(a, b, out, n);
#else
    detail::add_f32_scalar(a, b, out, n);
#endif
}

void sub_f32(const float* a, const float* b, float* out, size_t n) {
#if defined(__riscv_vector)
    detail::sub_f32_rvv(a, b, out, n);
#else
    detail::sub_f32_scalar(a, b, out, n);
#endif
}

void scale_f32(const float* a, float k, float* out, size_t n) {
#if defined(__riscv_vector)
    detail::scale_f32_rvv(a, k, out, n);
#else
    detail::scale_f32_scalar(a, k, out, n);
#endif
}

float dot_f32(const float* a, const float* b, size_t n) {
#if defined(__riscv_vector)
    return detail::dot_f32_rvv(a, b, n);
#else
    return detail::dot_f32_scalar(a, b, n);
#endif
}

float norm_l2_f32(const float* a, size_t n) {
#if defined(__riscv_vector)
    return detail::norm_l2_f32_rvv(a, n);
#else
    return detail::norm_l2_f32_scalar(a, n);
#endif
}

void matmul_f32(const float* A, const float* B, float* C, size_t m, size_t k, size_t n) {
#if defined(__riscv_vector)
    detail::matmul_f32_rvv(A, B, C, m, k, n);
#else
    detail::matmul_f32_scalar(A, B, C, m, k, n);
#endif
}

} // namespace rvv
