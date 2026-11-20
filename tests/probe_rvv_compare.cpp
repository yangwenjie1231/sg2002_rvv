// RVV 0.7.1 compare/mask semantics probe for SG2002 / C906 (GCC 10.2 T-Head).
#include <cstdio>
#include <cstdint>
#include <vector>
#include <riscv_vector.h>

static std::vector<float> g_vals;

static void dump_mask(const char* name, const float* vals, size_t vl) {
    std::vector<uint32_t> buf(64, 0xDEADBEEF);
    printf("%s vals: ", name);
    for (size_t i = 0; i < vl; ++i) printf("%.1f ", vals[i]);
    printf("\n");
    (void)buf;
}

#define SHOW(label, vec)                                                  \
    do {                                                                  \
        std::vector<uint32_t> b(64, 0xDEADBEEF);                          \
        vuint32m2_t t = vmerge_vvm_u32m2((vec), vmv_v_x_u32m2(1, vl),     \
                                         vmv_v_x_u32m2(0, vl), vl);       \
        vse32_v_u32m2(b.data(), t, vl);                                   \
        printf("%-22s -> ", label);                                       \
        for (size_t i = 0; i < vl; ++i) printf("%u ", b[i]);              \
        printf("\n");                                                     \
    } while (0)

int main() {
    float vals[8] = {100.0f, 250.0f, 255.0f, 249.0f, 250.5f, 0.0f, 251.0f, 240.0f};
    const size_t vl = vsetvl_e32m2(8);
    vfloat32m2_t c = vle32_v_f32m2(vals, vl);
    const float f = 250.0f;

    dump_mask("input", vals, vl);
    printf("expect >=250 : 0 1 1 0 1 0 1 0\n");
    printf("expect <  250: 1 0 0 1 0 1 0 1\n\n");

    SHOW("vmfge_vf(c,f)", vmfge_vf_f32m2_b16(c, f, vl));
    SHOW("vmfgt_vf(c,f)", vmfgt_vf_f32m2_b16(c, f, vl));
    SHOW("vmfle_vf(c,f)", vmfle_vf_f32m2_b16(c, f, vl));
    SHOW("vmflt_vf(c,f)", vmflt_vf_f32m2_b16(c, f, vl));
    SHOW("vmfne_vf(c,f)", vmfne_vf_f32m2_b16(c, f, vl));
    SHOW("vmfeq_vf(c,f)", vmfeq_vf_f32m2_b16(c, f, vl));

    printf("\n== vfmax/vfmin with constant ==\n");
    {
        vfloat32m2_t m = vfmax_vf_f32m2(c, 250.0f, vl);
        std::vector<float> out(64, -1.0f);
        vse32_v_f32m2(out.data(), m, vl);
        printf("vfmax_vf(c,250): ");
        for (size_t i = 0; i < vl; ++i) printf("%.1f ", out[i]);
        printf("\n");
    }

    printf("\n== reduction availability ==\n");
    {
        const size_t vlmax = vsetvlmax_e32m2();
        vfloat32m2_t v = vle32_v_f32m2(vals, vlmax < 8 ? vlmax : 8);
        vfloat32m1_t d = vfmv_v_f_f32m1(-1e30f, 1);
        vfloat32m1_t r = vfredmax_vs_f32m2_f32m1(d, v, d, vlmax < 8 ? vlmax : 8);
        printf("vfredmax = %.1f (expect 255.0)\n", vfmv_f_s_f32m1_f32(r));
        vfloat32m1_t d2 = vfmv_v_f_f32m1(1e30f, 1);
        vfloat32m1_t r2 = vfredmin_vs_f32m2_f32m1(d2, v, d2, vlmax < 8 ? vlmax : 8);
        printf("vfredmin = %.1f (expect 0.0)\n", vfmv_f_s_f32m1_f32(r2));
    }

    printf("\n== u32 reduce for counting ==\n");
    {
        const size_t vlmax = vsetvlmax_e32m2();
        vuint32m2_t ones = vmv_v_x_u32m2(1, vlmax);
        vuint32m1_t d = vmv_v_x_u32m1(0, 1);
        vuint32m1_t r = vredsum_vs_u32m2_u32m1(d, ones, d, vlmax);
        printf("vredsum all-ones = %u (expect %zu)\n", vmv_x_s_u32m1_u32(r), vlmax);
    }

    printf("\n== vfirst_m_b16 on a known mask ==\n");
    {
        vbool16_t gt = vmfgt_vf_f32m2_b16(c, f, vl);
        printf("vfirst_m_b16(vmfgt) = %ld (expect 1)\n", vfirst_m_b16(gt, vl));
        vbool16_t lt = vmflt_vf_f32m2_b16(c, f, vl);
        printf("vfirst_m_b16(vmflt) = %ld (expect 0)\n", vfirst_m_b16(lt, vl));
    }

    printf("\n== NaN detection candidates ==\n");
    {
        float withnan[8] = {1.0f, 2.0f, 0.0f/0.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        vfloat32m2_t vn = vle32_v_f32m2(withnan, vl);
        vbool16_t neq = vmfne_vv_f32m2_b16(vn, vn, vl);
        std::vector<uint32_t> b(64, 0xDEADBEEF);
        vuint32m2_t t = vmerge_vvm_u32m2(neq, vmv_v_x_u32m2(1, vl),
                                         vmv_v_x_u32m2(0, vl), vl);
        vse32_v_u32m2(b.data(), t, vl);
        printf("vmfne(vn,vn) bits: ");
        for (size_t i = 0; i < vl; ++i) printf("%u ", b[i]);
        printf("(expect 0 0 1 0 0 0 0 0, i.e. only NaN lane)\n");
        printf("vfirst_m_b16(vmfne) = %ld\n", vfirst_m_b16(neq, vl));
    }
    return 0;
}
