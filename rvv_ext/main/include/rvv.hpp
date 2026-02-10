#pragma once

#include <cstddef>

namespace rvv {

// Public API (always available, float32)
void add_f32(const float* a, const float* b, float* out, size_t n);
void sub_f32(const float* a, const float* b, float* out, size_t n);
void scale_f32(const float* a, float k, float* out, size_t n);
float dot_f32(const float* a, const float* b, size_t n);
float norm_l2_f32(const float* a, size_t n);

// Matrix multiplication: C = A @ B
// A: (m,k), B: (k,n), C: (m,n) all row-major contiguous.
void matmul_f32(const float* A, const float* B, float* C, size_t m, size_t k, size_t n);

// Implementation layer.
// - scalar implementations are always built
// - rvv implementations are built only when __riscv_vector is available
namespace detail {

void add_f32_scalar(const float* a, const float* b, float* out, size_t n);
void sub_f32_scalar(const float* a, const float* b, float* out, size_t n);
void scale_f32_scalar(const float* a, float k, float* out, size_t n);
float dot_f32_scalar(const float* a, const float* b, size_t n);
float norm_l2_f32_scalar(const float* a, size_t n);
void matmul_f32_scalar(const float* A, const float* B, float* C, size_t m, size_t k, size_t n);

#if defined(__riscv_vector)
void add_f32_rvv(const float* a, const float* b, float* out, size_t n);
void sub_f32_rvv(const float* a, const float* b, float* out, size_t n);
void scale_f32_rvv(const float* a, float k, float* out, size_t n);
float dot_f32_rvv(const float* a, const float* b, size_t n);
float norm_l2_f32_rvv(const float* a, size_t n);
void matmul_f32_rvv(const float* A, const float* B, float* C, size_t m, size_t k, size_t n);
#endif

} // namespace detail

} // namespace rvv
