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
/*
 * ⚠️ 2026-11-20 真机修复（MaixCAM / C906）：
 *
 * 手册《Configuration-Setting and vl Argument》：
 *   1. intrinsic "will only operate at most VLMAX element"，且 **只处理 vl 个元素**；
 *   2. 默认 **tail agnostic**：vl < VLMAX 时，超过 vl 的 lane 是未定义值。
 *
 * 原实现有三类错误（真机均复现）：
 *   · dot/norm_l2 的水平归约把 vl 硬编码成 4 ⇒ 只累加前 4 个 lane；
 *   · sum_f32/mean_std_f32 把 `vl` 元素的结果累加进跨块保持的向量累加器 ⇒
 *     最后一块不满时，acc 的 tail lane 被写成 agnostic，数据静默丢失；
 *   · max/min 用 `vmfne`+`vfirst` 做 NaN 检测，在真机上恒返回 NaN。
 *
 * ⇒ 统一改为：**跨迭代只用整块 vlmax，尾部走标量**；不依赖 agnostic lane。
 */
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
        const std::size_t block_end = std::min(n, i + kSumBlock);
        /*
         * 跨块累加只能用 **整 vlmax 块**：不满一块时 tail lane 是 agnostic，
         * 会被后续 vfadd 带进累加器。尾部的零头留给标量处理。
         */
        const std::size_t vec_end =
            i + ((block_end - i) / vlmax) * vlmax;
        vfloat32m4_t acc = vfmv_v_f_f32m4(0.0f, vlmax);
        while (i < vec_end) {
            acc = vfadd_vv_f32m4(acc, vle32_v_f32m4(a + i, vlmax), vlmax);
            i += vlmax;
        }
        total += static_cast<double>(reduce_f32m4(acc));
        // 尾部不足一个整块：标量收尾
        for (; i < block_end; ++i) {
            total += static_cast<double>(a[i]);
        }
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
        const std::size_t block_end = std::min(n, i + kSumBlock);
        // 同 sum_f32：跨迭代只用整块，尾部走标量，避免 agnostic lane 污染。
        const std::size_t vec_end = i + ((block_end - i) / vlmax) * vlmax;
        vfloat32m4_t acc = vfmv_v_f_f32m4(0.0f, vlmax);
        while (i < vec_end) {
            const vfloat32m4_t va = vle32_v_f32m4(a + i, vlmax);
            const vfloat32m4_t d = vfsub_vf_f32m4(va, mean_f, vlmax);
            acc = vfadd_vv_f32m4(acc, vfmul_vv_f32m4(d, d, vlmax), vlmax);
            i += vlmax;
        }
        sq += static_cast<double>(reduce_f32m4(acc));
        for (; i < block_end; ++i) {
            const double d = static_cast<double>(a[i]) - mean;
            sq += d * d;
        }
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
    /*
     * ⚠️ 2026-11-20 真机修复（MaixCAM / C906）：原实现用
     *   vfirst_m_b8(vmfne_vv_f32m4_b8(va, va, vl), vl) >= 0
     * 检测 NaN，但在真机上**恒为真** ⇒ max_f32/min_f32 对任何输入都返回 NaN。
     *
     * 这里改用不依赖 mask 语义的写法：向量只做 vfmax/vfmin 归并，
     * NaN 检测与水平归约全部走标量（标量侧 IEEE-754 语义明确）。
     */
    const std::size_t vlmax = vsetvlmax_e32m4();
    vfloat32m4_t acc = vfmv_v_f_f32m4(
        -std::numeric_limits<float>::infinity(), vlmax);
    std::size_t i = 0;
    for (; i + vlmax <= n; i += vlmax) {
        const vfloat32m4_t va = vle32_v_f32m4(a + i, vlmax);
        acc = vfmax_vv_f32m4(acc, va, vlmax);
    }
    std::vector<float> lanes(vlmax);
    vse32_v_f32m4(lanes.data(), acc, vlmax);
    float best = -std::numeric_limits<float>::infinity();
    for (float value : lanes) {
        if (std::isnan(value)) {
            return nan;
        }
        if (value > best) {
            best = value;
        }
    }
    for (; i < n; ++i) {
        if (std::isnan(a[i])) {
            return nan;
        }
        if (a[i] > best) {
            best = a[i];
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
    // 同 max_f32：向量只做归并，NaN 检测与水平归约走标量。
    const std::size_t vlmax = vsetvlmax_e32m4();
    vfloat32m4_t acc = vfmv_v_f_f32m4(
        std::numeric_limits<float>::infinity(), vlmax);
    std::size_t i = 0;
    for (; i + vlmax <= n; i += vlmax) {
        const vfloat32m4_t va = vle32_v_f32m4(a + i, vlmax);
        acc = vfmin_vv_f32m4(acc, va, vlmax);
    }
    std::vector<float> lanes(vlmax);
    vse32_v_f32m4(lanes.data(), acc, vlmax);
    float best = std::numeric_limits<float>::infinity();
    for (float value : lanes) {
        if (std::isnan(value)) {
            return nan;
        }
        if (value < best) {
            best = value;
        }
    }
    for (; i < n; ++i) {
        if (std::isnan(a[i])) {
            return nan;
        }
        if (a[i] < best) {
            best = a[i];
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
