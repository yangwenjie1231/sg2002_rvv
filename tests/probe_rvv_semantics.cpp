// RVV 0.7.1 primitive probe for SG2002 / C906.
// Cross-compile, push to device, run. Prints raw facts so fixes are evidence-based.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <riscv_vector.h>

static void probe_vlmax() {
    printf("== VLMAX ==\n");
    printf("e8m1=%zu e8m4=%zu e8m8=%zu\n",
           vsetvlmax_e8m1(), vsetvlmax_e8m4(), vsetvlmax_e8m8());
    printf("e16m2=%zu e16m4=%zu\n", vsetvlmax_e16m2(), vsetvlmax_e16m4());
    printf("e32m1=%zu e32m2=%zu e32m4=%zu e32m8=%zu\n",
           vsetvlmax_e32m1(), vsetvlmax_e32m2(), vsetvlmax_e32m4(), vsetvlmax_e32m8());
}

static void probe_widen_u8() {
    printf("== widen u8 -> u16 -> f32 ==\n");
    uint8_t src[16];
    for (int i = 0; i < 16; ++i) src[i] = (uint8_t)(i * 16);
    const size_t vl = vsetvl_e8m1(16);
    vuint8m1_t v = vle8_v_u8m1(src, vl);
    vuint16m2_t w = vwaddu_vx_u16m2(v, 0, vl);
    std::vector<uint16_t> wbuf(64, 0xAAAA);
    vse16_v_u16m2(wbuf.data(), w, vl);
    printf("vl=%zu u16[0..7]=", vl);
    for (int i = 0; i < 8; ++i) printf("%u ", (unsigned)wbuf[i]);
    printf("\n");
    vfloat32m4_t f = vfwcvt_f_xu_v_f32m4(w, vl);
    std::vector<float> fbuf(128, -777.0f);
    vse32_v_f32m4(fbuf.data(), f, vl);
    printf("f32[0..7]=");
    for (int i = 0; i < 8; ++i) printf("%.1f ", fbuf[i]);
    printf("\n");
}

static void probe_stride3() {
    printf("== vlse8 stride=3 (RGB deinterleave) ==\n");
    // 16 pixels: R=i*10+1, G=i*10+2, B=i*10+3
    uint8_t rgb[16 * 3];
    for (int i = 0; i < 16; ++i) {
        rgb[i * 3 + 0] = (uint8_t)(i * 10 + 1);
        rgb[i * 3 + 1] = (uint8_t)(i * 10 + 2);
        rgb[i * 3 + 2] = (uint8_t)(i * 10 + 3);
    }
    const size_t vl = vsetvl_e8m1(16);
    vuint8m1_t r = vlse8_v_u8m1(rgb + 0, 3, vl);
    vuint8m1_t g = vlse8_v_u8m1(rgb + 1, 3, vl);
    vuint8m1_t b = vlse8_v_u8m1(rgb + 2, 3, vl);
    std::vector<uint8_t> rb(64, 0xEE), gb(64, 0xEE), bb(64, 0xEE);
    vse8_v_u8m1(rb.data(), r, vl);
    vse8_v_u8m1(gb.data(), g, vl);
    vse8_v_u8m1(bb.data(), b, vl);
    printf("vl=%zu R[0..7]=", vl);
    for (int i = 0; i < 8; ++i) printf("%u ", rb[i]);
    printf("\n        G[0..7]=");
    for (int i = 0; i < 8; ++i) printf("%u ", gb[i]);
    printf("\n        B[0..7]=");
    for (int i = 0; i < 8; ++i) printf("%u ", bb[i]);
    printf("\n");
}

static void probe_mask_merge() {
    printf("== vmfge + vmerge operand order ==\n");
    float vals[8] = {100.0f, 250.0f, 255.0f, 249.0f, 250.5f, 0.0f, 251.0f, 240.0f};
    const size_t vl = vsetvl_e32m2(8);
    vfloat32m2_t c = vle32_v_f32m2(vals, vl);
    vbool16_t m = vmfge_vf_f32m2_b16(c, 250.0f, vl);
    // No vget for masks in this toolchain; materialise via vmerge into u32.
    vuint32m2_t probe_one = vmv_v_x_u32m2(1, vl);
    vuint32m2_t probe_zero = vmv_v_x_u32m2(0, vl);
    vuint32m2_t maskbits = vmerge_vvm_u32m2(m, probe_one, probe_zero, vl);
    std::vector<uint32_t> mb(64, 0xDEADBEEF);
    vse32_v_u32m2(mb.data(), maskbits, vl);
    printf("mask bits(0..7)= ");
    for (int i = 0; i < 8; ++i) printf("%u ", mb[i]);
    printf("\n");
    vuint32m2_t one = vmv_v_x_u32m2(1, vl);
    vuint32m2_t zero = vmv_v_x_u32m2(0, vl);
    // order A: vmerge(mask, one, zero) ; order B: vmerge(mask, zero, one)
    vuint32m2_t a = vmerge_vvm_u32m2(m, one, zero, vl);
    vuint32m2_t b = vmerge_vvm_u32m2(m, zero, one, vl);
    std::vector<uint32_t> ab(64, 0xDEADBEEF), bb(64, 0xDEADBEEF);
    vse32_v_u32m2(ab.data(), a, vl);
    vse32_v_u32m2(bb.data(), b, vl);
    printf("A vmerge(m,one,zero)= ");
    for (int i = 0; i < 8; ++i) printf("%u ", ab[i]);
    printf("\nB vmerge(m,zero,one)= ");
    for (int i = 0; i < 8; ++i) printf("%u ", bb[i]);
    printf("\n");
}

static void probe_reduce_vl() {
    printf("== reduction: vl vs vlmax, tail agnostic ==\n");
    const size_t vlmax = vsetvlmax_e32m2();
    printf("vlmax_e32m2=%zu\n", vlmax);
    // ones vector of length vlmax
    std::vector<float> ones(vlmax * 2, 1.0f);
    vfloat32m2_t v = vle32_v_f32m2(ones.data(), vlmax);
    // reduce with vl = 1 vs vl = vlmax
    for (size_t rvl : {size_t(1), size_t(2), vlmax}) {
        vfloat32m1_t d = vfmv_v_f_f32m1(0.0f, 1);
        vfloat32m1_t r = vfredusum_vs_f32m2_f32m1(d, v, d, rvl);
        printf("vfredusum(vl=%zu of ones) = %.1f\n", rvl, vfmv_f_s_f32m1_f32(r));
    }
    // Does vfadd with vl<vlmax clobber tail lanes of the destination?
    printf("-- tail agnostic test: acc=1, add ones with vl=1 --\n");
    vfloat32m2_t acc = vfmv_v_f_f32m2(1.0f, vlmax);
    vfloat32m2_t addend = vfmv_v_f_f32m2(100.0f, vlmax);
    acc = vfadd_vv_f32m2(acc, addend, 1);  // vl=1
    std::vector<float> out(64, -1.0f);
    vse32_v_f32m2(out.data(), acc, vlmax);
    printf("acc lanes after vl=1 add: ");
    for (size_t i = 0; i < vlmax; ++i) printf("%.0f ", out[i]);
    printf("\n(expect 101 in lane0; other lanes 1 if undisturbed, garbage if agnostic)\n");
}

static void probe_dot_style() {
    printf("== incremental accumulate then reduce ==\n");
    const size_t vlmax = vsetvlmax_e32m2();
    float data[20];
    for (int i = 0; i < 20; ++i) data[i] = 1.0f;
    // emulate upstream pattern: loop with vl = vsetvl(n-i), accumulate
    vfloat32m2_t acc = vfmv_v_f_f32m2(0.0f, vlmax);
    size_t i = 0;
    while (i < 20) {
        size_t vl = vsetvl_e32m2(20 - i);
        vfloat32m2_t v = vle32_v_f32m2(data + i, vl);
        acc = vfadd_vv_f32m2(acc, v, vl);
        i += vl;
    }
    vfloat32m1_t d = vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t r = vfredusum_vs_f32m2_f32m1(d, acc, d, vlmax);
    printf("sum(20 ones) incremental+vlmax reduce = %.2f (expect 20)\n",
           vfmv_f_s_f32m1_f32(r));
    // safe pattern: full vlmax blocks + scalar tail
    vfloat32m2_t acc2 = vfmv_v_f_f32m2(0.0f, vlmax);
    size_t j = 0;
    for (; j + vlmax <= 20; j += vlmax) {
        vfloat32m2_t v = vle32_v_f32m2(data + j, vlmax);
        acc2 = vfadd_vv_f32m2(acc2, v, vlmax);
    }
    vfloat32m1_t d2 = vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t r2 = vfredusum_vs_f32m2_f32m1(d2, acc2, d2, vlmax);
    float s2 = vfmv_f_s_f32m1_f32(r2);
    for (; j < 20; ++j) s2 += data[j];
    printf("sum(20 ones) full-block+scalar-tail = %.2f (expect 20)\n", s2);
}

int main() {
    probe_vlmax();
    probe_widen_u8();
    probe_stride3();
    probe_mask_merge();
    probe_reduce_vl();
    probe_dot_style();
    return 0;
}
