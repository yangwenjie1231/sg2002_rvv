#include "rvv.hpp"

#if defined(__riscv_vector)

#include <riscv_vector.h>
#include <cmath>

namespace rvv::detail {

void add_f32_rvv(const float* a, const float* b, float* out, size_t n) {
    while (n > 0) {
        size_t vl = vsetvl_e32m8(n);
        vfloat32m8_t va = vle32_v_f32m8(a, vl);
        vfloat32m8_t vb = vle32_v_f32m8(b, vl);
        vfloat32m8_t vc = vfadd_vv_f32m8(va, vb, vl);
        vse32_v_f32m8(out, vc, vl);
        a += vl;
        b += vl;
        out += vl;
        n -= vl;
    }
}

void sub_f32_rvv(const float* a, const float* b, float* out, size_t n) {
    while (n > 0) {
        size_t vl = vsetvl_e32m8(n);
        vfloat32m8_t va = vle32_v_f32m8(a, vl);
        vfloat32m8_t vb = vle32_v_f32m8(b, vl);
        vfloat32m8_t vc = vfsub_vv_f32m8(va, vb, vl);
        vse32_v_f32m8(out, vc, vl);
        a += vl;
        b += vl;
        out += vl;
        n -= vl;
    }
}

void scale_f32_rvv(const float* a, float k, float* out, size_t n) {
    while (n > 0) {
        size_t vl = vsetvl_e32m8(n);
        vfloat32m8_t va = vle32_v_f32m8(a, vl);
        vfloat32m8_t vc = vfmul_vf_f32m8(va, k, vl);
        vse32_v_f32m8(out, vc, vl);
        a += vl;
        out += vl;
        n -= vl;
    }
}

float dot_f32_rvv(const float* a, const float* b, size_t n) {
    double acc = 0.0;
    while (n > 0) {
        size_t vl = vsetvl_e32m8(n);
        vfloat32m8_t va = vle32_v_f32m8(a, vl);
        vfloat32m8_t vb = vle32_v_f32m8(b, vl);
        vfloat32m8_t vm = vfmul_vv_f32m8(va, vb, vl);

        alignas(64) float tmp[256];
        vse32_v_f32m8(tmp, vm, vl);
        for (size_t i = 0; i < vl; ++i) acc += tmp[i];

        a += vl;
        b += vl;
        n -= vl;
    }
    return static_cast<float>(acc);
}

float norm_l2_f32_rvv(const float* a, size_t n) {
    return std::sqrt(dot_f32_rvv(a, a, n));
}

void matmul_f32_rvv(const float* A, const float* B, float* C, size_t m, size_t k, size_t n) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) C[i * n + j] = 0.0f;

        for (size_t kk = 0; kk < k; ++kk) {
            float a_ik = A[i * k + kk];
            const float* b_row = B + kk * n;
            float* c_row = C + i * n;

            size_t remaining = n;
            while (remaining > 0) {
                size_t vl = vsetvl_e32m8(remaining);
                vfloat32m8_t vb = vle32_v_f32m8(b_row, vl);
                vfloat32m8_t vc = vle32_v_f32m8(c_row, vl);
                vc = vfmacc_vf_f32m8(vc, a_ik, vb, vl);
                vse32_v_f32m8(c_row, vc, vl);

                b_row += vl;
                c_row += vl;
                remaining -= vl;
            }
        }
    }
}

} // namespace rvv::detail

#endif // __riscv_vector
