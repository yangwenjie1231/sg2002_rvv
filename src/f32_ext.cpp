#include "rvv.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

namespace rvv::core {
namespace {

#if defined(__riscv_vector)
// Do not rely on float64 vector intrinsics: C906 deployments/toolchains
// differ on 64-bit vector element support. Accumulate float32 vector
// partials in bounded blocks and add each block to a double scalar.
constexpr std::size_t kSumBlock = 4096;

float reduce_f32m4(const vfloat32m4_t v) {
    const std::size_t vlmax = vsetvlmax_e32m4();
    vfloat32m1_t acc = vfmv_v_f_f32m1(0.0f, 1);
    acc = vfredusum_vs_f32m4_f32m1(acc, v, acc, vlmax);
    return vfmv_f_s_f32m1_f32(acc);
}
#endif

}  // namespace

void abs_f32(const float* a, float* b, std::size_t n) {
#if defined(__riscv_vector)
    std::size_t i = 0;
    while (i < n) {
        const std::size_t vl = vsetvl_e32m4(n - i);
        const vfloat32m4_t va = vle32_v_f32m4(a + i, vl);
        vse32_v_f32m4(b + i, vfabs_v_f32m4(va, vl), vl);
        i += vl;
    }
#else
    for (std::size_t i = 0; i < n; ++i) {
        b[i] = std::fabs(a[i]);
    }
#endif
}

double sum_f32(const float* a, std::size_t n) {
    double total = 0.0;
#if defined(__riscv_vector)
    const std::size_t vlmax = vsetvlmax_e32m4();
    std::size_t i = 0;
    while (i < n) {
        const std::size_t block_end =
            std::min(n, i + kSumBlock);
        vfloat32m4_t acc = vfmv_v_f_f32m4(0.0f, vlmax);
        while (i < block_end) {
            const std::size_t vl = vsetvl_e32m4(block_end - i);
            acc = vfadd_vv_f32m4(
                acc, vle32_v_f32m4(a + i, vl), vl);
            i += vl;
        }
        total += static_cast<double>(reduce_f32m4(acc));
    }
#else
    for (std::size_t i = 0; i < n; ++i) {
        total += static_cast<double>(a[i]);
    }
#endif
    return total;
}

std::pair<double, double> mean_std_f32(const float* a, std::size_t n) {
    if (n == 0) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return {nan, nan};
    }

    const double mean = sum_f32(a, n) / static_cast<double>(n);
    double sq = 0.0;

#if defined(__riscv_vector)
    const std::size_t vlmax = vsetvlmax_e32m4();
    const float mean_f = static_cast<float>(mean);
    std::size_t i = 0;
    while (i < n) {
        const std::size_t block_end =
            std::min(n, i + kSumBlock);
        vfloat32m4_t acc = vfmv_v_f_f32m4(0.0f, vlmax);
        while (i < block_end) {
            const std::size_t vl = vsetvl_e32m4(block_end - i);
            const vfloat32m4_t va = vle32_v_f32m4(a + i, vl);
            const vfloat32m4_t d = vfsub_vf_f32m4(va, mean_f, vl);
            acc = vfadd_vv_f32m4(
                acc, vfmul_vv_f32m4(d, d, vl), vl);
            i += vl;
        }
        sq += static_cast<double>(reduce_f32m4(acc));
    }
#else
    for (std::size_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(a[i]) - mean;
        sq += d * d;
    }
#endif

    double var = sq / static_cast<double>(n);
    if (var < 0.0) {
        var = 0.0;
    }
    return {mean, std::sqrt(var)};
}

void mul_f32(const float* a, const float* b, float* c, std::size_t n) {
#if defined(__riscv_vector)
    std::size_t i = 0;
    while (i < n) {
        const std::size_t vl = vsetvl_e32m4(n - i);
        const vfloat32m4_t va = vle32_v_f32m4(a + i, vl);
        const vfloat32m4_t vb = vle32_v_f32m4(b + i, vl);
        vse32_v_f32m4(c + i, vfmul_vv_f32m4(va, vb, vl), vl);
        i += vl;
    }
#else
    for (std::size_t i = 0; i < n; ++i) {
        c[i] = a[i] * b[i];
    }
#endif
}

void add_scalar_f32(const float* a, float k, float* b, std::size_t n) {
#if defined(__riscv_vector)
    std::size_t i = 0;
    while (i < n) {
        const std::size_t vl = vsetvl_e32m4(n - i);
        const vfloat32m4_t va = vle32_v_f32m4(a + i, vl);
        vse32_v_f32m4(b + i, vfadd_vf_f32m4(va, k, vl), vl);
        i += vl;
    }
#else
    for (std::size_t i = 0; i < n; ++i) {
        b[i] = a[i] + k;
    }
#endif
}

void clip_f32(const float* a, float lo, float hi, float* b, std::size_t n) {
    if (lo > hi) {
        std::swap(lo, hi);
    }
#if defined(__riscv_vector)
    const std::size_t vlmax = vsetvlmax_e32m4();
    const vfloat32m4_t vlo = vfmv_v_f_f32m4(lo, vlmax);
    const vfloat32m4_t vhi = vfmv_v_f_f32m4(hi, vlmax);
    std::size_t i = 0;
    while (i < n) {
        const std::size_t vl = vsetvl_e32m4(n - i);
        vfloat32m4_t va = vle32_v_f32m4(a + i, vl);
        // Comparisons are false for NaN, so NaN is preserved by both merges.
        const vbool8_t below = vmflt_vf_f32m4_b8(va, lo, vl);
        va = vmerge_vvm_f32m4(below, va, vlo, vl);
        const vbool8_t above = vmfgt_vf_f32m4_b8(va, hi, vl);
        va = vmerge_vvm_f32m4(above, va, vhi, vl);
        vse32_v_f32m4(b + i, va, vl);
        i += vl;
    }
#else
    for (std::size_t i = 0; i < n; ++i) {
        const float v = a[i];
        b[i] = v < lo ? lo : (v > hi ? hi : v);
    }
#endif
}

float max_f32(const float* a, std::size_t n) {
    if (n == 0) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const float nan = std::numeric_limits<float>::quiet_NaN();
#if defined(__riscv_vector)
    const std::size_t vlmax = vsetvlmax_e32m4();
    vfloat32m4_t acc = vfmv_v_f_f32m4(
        -std::numeric_limits<float>::infinity(), vlmax);
    std::size_t i = 0;
    while (i < n) {
        const std::size_t vl = vsetvl_e32m4(n - i);
        const vfloat32m4_t va = vle32_v_f32m4(a + i, vl);
        // vmfne is true for unordered (NaN) lanes; propagate NumPy max()
        // semantics instead of RISC-V maxNum semantics.
        if (vfirst_m_b8(vmfne_vv_f32m4_b8(va, va, vl), vl) >= 0) {
            return nan;
        }
        acc = vfmax_vv_f32m4(acc, va, vl);
        i += vl;
    }
    // Horizontal reduction is done over VLMAX lanes in scalar code.  This
    // avoids depending on vfredmax while keeping the expensive pairwise
    // reduction vectorized.
    std::vector<float> lanes(vlmax);
    vse32_v_f32m4(lanes.data(), acc, vlmax);
    float best = -std::numeric_limits<float>::infinity();
    for (float value : lanes) {
        if (value > best) {
            best = value;
        }
    }
    return best;
#else
    float best = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isnan(a[i])) {
            return nan;
        }
        if (a[i] > best) {
            best = a[i];
        }
    }
    return best;
#endif
}

float min_f32(const float* a, std::size_t n) {
    if (n == 0) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const float nan = std::numeric_limits<float>::quiet_NaN();
#if defined(__riscv_vector)
    const std::size_t vlmax = vsetvlmax_e32m4();
    vfloat32m4_t acc = vfmv_v_f_f32m4(
        std::numeric_limits<float>::infinity(), vlmax);
    std::size_t i = 0;
    while (i < n) {
        const std::size_t vl = vsetvl_e32m4(n - i);
        const vfloat32m4_t va = vle32_v_f32m4(a + i, vl);
        if (vfirst_m_b8(vmfne_vv_f32m4_b8(va, va, vl), vl) >= 0) {
            return nan;
        }
        acc = vfmin_vv_f32m4(acc, va, vl);
        i += vl;
    }
    std::vector<float> lanes(vlmax);
    vse32_v_f32m4(lanes.data(), acc, vlmax);
    float best = std::numeric_limits<float>::infinity();
    for (float value : lanes) {
        if (value < best) {
            best = value;
        }
    }
    return best;
#else
    float best = std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isnan(a[i])) {
            return nan;
        }
        if (a[i] < best) {
            best = a[i];
        }
    }
    return best;
#endif
}

}  // namespace rvv::core
