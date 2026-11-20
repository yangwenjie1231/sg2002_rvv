#include "rvv.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

namespace rvv::core {
namespace {

struct Roi {
    std::size_t x0 = 0;
    std::size_t y0 = 0;
    std::size_t width = 0;
    std::size_t height = 0;
    std::size_t step_x = 0;
    std::size_t step_y = 0;
    std::size_t count_x = 0;
    std::size_t count_y = 0;
};

inline double gray_at(const std::uint8_t* row, std::size_t channels) {
    if (channels == 1) {
        return static_cast<double>(row[0]);
    }
    return (static_cast<double>(row[0]) +
            static_cast<double>(row[1]) +
            static_cast<double>(row[2])) / 3.0;
}

Roi make_roi(std::size_t width,
             std::size_t height,
             double roi_fraction,
             std::size_t step_x,
             std::size_t step_y,
             std::size_t offset) {
    Roi roi;
    if (width < 4 || height < 4 || offset == 0) {
        return roi;
    }

    const double fraction = std::min(std::max(roi_fraction, 0.1), 1.0);
    roi.width = std::max<std::size_t>(
        4, static_cast<std::size_t>(static_cast<double>(width) * fraction));
    roi.height = std::max<std::size_t>(
        4, static_cast<std::size_t>(static_cast<double>(height) * fraction));
    roi.x0 = (width - roi.width) / 2;
    roi.y0 = (height - roi.height) / 2;

    roi.step_x = step_x == 0 ? std::max<std::size_t>(2, roi.width / 100) : step_x;
    roi.step_y = step_y == 0 ? std::max<std::size_t>(2, roi.height / 100) : step_y;

    if (roi.width <= offset || roi.height <= offset) {
        return roi;
    }
    roi.count_x = (roi.width - offset - 1) / roi.step_x + 1;
    roi.count_y = (roi.height - offset - 1) / roi.step_y + 1;
    return roi;
}

GradientStats gradient_stats_u8_scalar(const std::uint8_t* data,
                                       std::size_t stride,
                                       const Roi& roi,
                                       std::size_t offset,
                                       std::size_t channels) {
    GradientStats stats;
    for (std::size_t iy = 0; iy < roi.count_y; ++iy) {
        const std::size_t y = roi.y0 + iy * roi.step_y;
        const std::uint8_t* row_y = data + y * stride;
        const std::uint8_t* row_y_offset = row_y + offset * stride;
        for (std::size_t ix = 0; ix < roi.count_x; ++ix) {
            const std::size_t x = roi.x0 + ix * roi.step_x;
            const std::uint8_t* px = row_y + x * channels;
            const std::uint8_t* px_offset = px + offset * channels;
            const std::uint8_t* py_offset = row_y_offset + x * channels;

            const double c = gray_at(px, channels);
            const double cx = gray_at(px_offset, channels);
            const double cy = gray_at(py_offset, channels);
            const double dx = cx - c;
            const double dy = cy - c;

            stats.grad_sum += dx * dx + dy * dy;
            stats.lum_sum += c;
            stats.lum_sq_sum += c * c;
            ++stats.count;
            if (c >= 250.0) {
                ++stats.saturated_count;
            }
        }
    }
    return stats;
}

#if defined(__riscv_vector)

/*
 * ⚠️ 2026-11-20 真机规则（详见 docs/rvv071-device-semantics.md）：
 *
 * 1. tail lane 是 **agnostic**：`vl < vlmax` 时越界 lane 被写成未定义值。
 *    ⇒ 跨迭代向量累加器只能吃 **整 vlmax 块**，剩余元素走标量。
 * 2. 归约（vfredsum 等）的 vl 必须是 `vsetvlmax_*()`，不能用常数或 vsetvl 结果。
 * 3. `vmf*_vf`（标量与向量比较）在这套工具链上**语义错误**，禁止使用；
 *    `vfirst_m` / `vmfne_vv` 做判断也不可靠。
 *    ⇒ 饱和计数这类带比较的统计一律在**标量侧**做。
 */

// Convert one source row (gray or RGB888) to a tightly packed float row.
// RGB888 uses (R+G+B)/3.0 to match the firmware's Python reference.
// Only whole vlmax blocks are vectorised here; the caller handles the tail.
void gray_row_to_float_rvv(const std::uint8_t* src,
                           std::size_t width,
                           std::size_t channels,
                           float* dst) {
    const std::size_t vlmax8 = vsetvlmax_e8m1();   // C906: 16
    const std::size_t vec_width = (width / vlmax8) * vlmax8;
    std::size_t i = 0;
    while (i < vec_width) {
        const std::size_t vl = vlmax8;
        if (channels == 1) {
            const vuint8m1_t v = vle8_v_u8m1(src + i, vl);
            const vuint16m2_t w = vwaddu_vx_u16m2(v, 0, vl);
            const vfloat32m4_t f = vfwcvt_f_xu_v_f32m4(w, vl);
            vse32_v_f32m4(dst + i, f, vl);
        } else {
            const vuint8m1_t r = vlse8_v_u8m1(src + i * 3 + 0, 3, vl);
            const vuint8m1_t g = vlse8_v_u8m1(src + i * 3 + 1, 3, vl);
            const vuint8m1_t b = vlse8_v_u8m1(src + i * 3 + 2, 3, vl);
            const vuint16m2_t wr = vwaddu_vx_u16m2(r, 0, vl);
            const vuint16m2_t wg = vwaddu_vx_u16m2(g, 0, vl);
            const vuint16m2_t wb = vwaddu_vx_u16m2(b, 0, vl);
            const vfloat32m4_t fr = vfwcvt_f_xu_v_f32m4(wr, vl);
            const vfloat32m4_t fg = vfwcvt_f_xu_v_f32m4(wg, vl);
            const vfloat32m4_t fb = vfwcvt_f_xu_v_f32m4(wb, vl);
            const vfloat32m4_t sum = vfadd_vv_f32m4(
                vfadd_vv_f32m4(fr, fg, vl), fb, vl);
            const vfloat32m4_t gray = vfmul_vf_f32m4(
                sum, 1.0f / 3.0f, vl);
            vse32_v_f32m4(dst + i, gray, vl);
        }
        i += vl;
    }
    // Tail elements: scalar, so no agnostic lane ever enters the pipeline.
    for (; i < width; ++i) {
        if (channels == 1) {
            dst[i] = static_cast<float>(src[i]);
        } else {
            dst[i] = (static_cast<float>(src[i * 3 + 0]) +
                      static_cast<float>(src[i * 3 + 1]) +
                      static_cast<float>(src[i * 3 + 2])) / 3.0f;
        }
    }
}

float reduce_f32(const vfloat32m2_t v) {
    const std::size_t vlmax = vsetvlmax_e32m2();
    vfloat32m1_t acc = vfmv_v_f_f32m1(0.0f, 1);
    acc = vfredusum_vs_f32m2_f32m1(acc, v, acc, vlmax);
    return vfmv_f_s_f32m1_f32(acc);
}

GradientStats gradient_stats_u8_rvv(const std::uint8_t* data,
                                    std::size_t stride,
                                    const Roi& roi,
                                    std::size_t offset,
                                    std::size_t channels) {
    GradientStats stats;
    std::vector<float> row_c(roi.width);
    std::vector<float> row_offset(roi.width);

    const std::size_t vlmax = vsetvlmax_e32m2();  // C906: 8
    vfloat32m2_t grad_acc = vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t lum_acc = vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t lum_sq_acc = vfmv_v_f_f32m2(0.0f, vlmax);
    const std::ptrdiff_t sample_stride =
        static_cast<std::ptrdiff_t>(roi.step_x * sizeof(float));
    // 每行可整块向量化的采样点数量；零头走标量，避免 agnostic 尾 lane。
    const std::size_t vec_count = (roi.count_x / vlmax) * vlmax;

    for (std::size_t iy = 0; iy < roi.count_y; ++iy) {
        const std::size_t y = roi.y0 + iy * roi.step_y;
        const std::uint8_t* src_c =
            data + y * stride + roi.x0 * channels;
        const std::uint8_t* src_y =
            data + (y + offset) * stride + roi.x0 * channels;
        gray_row_to_float_rvv(src_c, roi.width, channels, row_c.data());
        gray_row_to_float_rvv(src_y, roi.width, channels, row_offset.data());

        // 整块部分：向量累加（只喂 vlmax，累加器不会被写坏）
        std::size_t ix = 0;
        while (ix < vec_count) {
            const std::size_t x = ix * roi.step_x;
            const float* base = row_c.data() + x;
            const vfloat32m2_t c =
                vlse32_v_f32m2(base, sample_stride, vlmax);
            const vfloat32m2_t cx =
                vlse32_v_f32m2(base + offset, sample_stride, vlmax);
            const vfloat32m2_t cy =
                vlse32_v_f32m2(row_offset.data() + x, sample_stride, vlmax);

            const vfloat32m2_t dx = vfsub_vv_f32m2(cx, c, vlmax);
            const vfloat32m2_t dy = vfsub_vv_f32m2(cy, c, vlmax);
            grad_acc = vfadd_vv_f32m2(
                grad_acc,
                vfadd_vv_f32m2(vfmul_vv_f32m2(dx, dx, vlmax),
                               vfmul_vv_f32m2(dy, dy, vlmax), vlmax),
                vlmax);
            lum_acc = vfadd_vv_f32m2(lum_acc, c, vlmax);
            lum_sq_acc = vfadd_vv_f32m2(
                lum_sq_acc, vfmul_vv_f32m2(c, c, vlmax), vlmax);
            ix += vlmax;
        }

        // 行内零头 + 饱和计数：全部标量（vmf*_vf 在真机语义错误）
        for (std::size_t k = 0; k < roi.count_x; ++k) {
            const std::size_t x = k * roi.step_x;
            const double c = static_cast<double>(row_c[x]);
            if (c >= 250.0) {
                ++stats.saturated_count;
            }
            if (k < vec_count) {
                continue;  // 梯度/亮度已由向量块累加
            }
            const double cx = static_cast<double>(row_c[x + offset]);
            const double cy = static_cast<double>(row_offset[x]);
            const double dx = cx - c;
            const double dy = cy - c;
            stats.grad_sum += dx * dx + dy * dy;
            stats.lum_sum += c;
            stats.lum_sq_sum += c * c;
        }
    }

    stats.grad_sum += static_cast<double>(reduce_f32(grad_acc));
    stats.lum_sum += static_cast<double>(reduce_f32(lum_acc));
    stats.lum_sq_sum += static_cast<double>(reduce_f32(lum_sq_acc));
    stats.count = static_cast<std::uint64_t>(roi.count_x) *
                  static_cast<std::uint64_t>(roi.count_y);
    return stats;
}

#endif  // defined(__riscv_vector)

}  // namespace

GradientStats gradient_stats_u8(const std::uint8_t* data,
                                std::size_t width,
                                std::size_t height,
                                std::size_t stride,
                                double roi_fraction,
                                std::size_t step_x,
                                std::size_t step_y,
                                std::size_t offset,
                                std::size_t channels) {
    const Roi roi = make_roi(width, height, roi_fraction, step_x, step_y, offset);
    if (data == nullptr || roi.count_x == 0 || roi.count_y == 0) {
        return GradientStats{};
    }
    if (channels != 1 && channels != 3) {
        return GradientStats{};
    }

#if defined(__riscv_vector)
    return gradient_stats_u8_rvv(data, stride, roi, offset, channels);
#else
    return gradient_stats_u8_scalar(data, stride, roi, offset, channels);
#endif
}

}  // namespace rvv::core
