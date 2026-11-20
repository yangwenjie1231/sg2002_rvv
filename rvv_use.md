# RVV 指令集加速运行效率，比如模型预处理/后处理

> **作者**：neucrack  
> **分类**：[知识 & 开发](/c/150) > [MCU SOC](/c/17) > [通用](/c/150)  
> **阅读**：2904　**点赞**：3  
> **版权**：[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.zh)  
> **创建**：2024-10-11　**更新**：2024-10-15  
> **原文链接（持续更新）**：https://neucrack.com/p/551  

---

## 资源

- [玄铁 C906 使用的 RVV 0.7.1 内建函数手册（PDF）](https://occ-oss-prod.oss-cn-hangzhou.aliyuncs.com/resource//1659318003104/Xuantie+900+Series+RVV-0.7.1+Intrinsic+Manual.pdf)

> ⚠️ 注意：RVV 不同版本不兼容（如 RVV 0.7.1 与 RVV 1.0.0），使用时务必确认芯片支持的具体版本。

---

## 使用场景

RVV（RISC-V Vector Extension）是一种矢量加速指令集，适用于：

- **矢量运算** 或 **数据并行运算**
- **批量化计算**：单条指令处理多个数据，类似 OpenMP 的多核并行思想
- **加速 for 循环**：减少指令数量，提升执行效率
- **优化内存拷贝**：例如 HWC → CHW 格式转换可使用 RVV 批量拷贝，比逐元素 for 循环更快

只要芯片支持的 RVV 指令集中包含所需操作（如加减乘除、逻辑运算等），即可加速。对于复杂函数（如 `tanh`），可基于基本 RVV 指令组合实现（参考 [ncnn 的 RVV tanh 实现](https://github.com/Tencent/ncnn/blob/9b5f6a39b4a4962accaad58caa771487f61f732a/src/layer/riscv/rvv_mathfun.h#L303)）。

---

## 例子：RVV 加速 `tanh` 批量计算

```cpp
#if __riscv_vector
int n = size;
while (n > 0) {
    size_t vl = vsetvl_e32m8(n);               // 设置向量长度
    vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);  // 加载数据到向量寄存器
    _p = tanh_ps(_p, vl);                      // 批量计算 tanh
    vse32_v_f32m8(ptr, _p, vl);                // 存回内存
    ptr += vl;
    n -= vl;
}
#else  // __riscv_vector
for (int i = 0; i < size; i++) {
    *ptr = tanh(*ptr);
    ptr++;
}
#endif  // __riscv_vector
```

### 关键说明：

- `vsetvl_e32m8(n)`：设置元素宽度为 32 位（SEW=32），LMUL=8。
  - 若硬件 VLEN=128，则一次最多处理 `min(n, 128/32 * 8) = min(n, 32)` 个 float32 元素。
- `vle32_v_f32m8` / `vse32_v_f32m8`：批量加载/存储。
- 即使只处理 18 个元素，使用 `m8` 仍只需一条指令（硬件自动掩码无效元素），效率高于分两次（如 m4 + m1）。

---

## 用 RVV 加速模型输入预处理 `(x - mean) * scale`

典型图像预处理：对每个像素执行 `(pixel - mean) * scale`。  
在 SG2002（C906）上，**从 9~14ms 降至 1~2ms**，提速显著。

### 原始 C++ 实现（HWC → CHW + 归一化）：
```cpp
for (int i = 0; i < img_h * img_w; ++i) {
    *ptr_ch0 = ((float)*p       - mean[0]) * scales[0];
    *ptr_ch1 = ((float)*(p + 1) - mean[1]) * scales[1];
    *ptr_ch2 = ((float)*(p + 2) - mean[2]) * scales[2];
    ++ptr_ch0; ++ptr_ch1; ++ptr_ch2;
    p += 3;
}
```

### RVV 加速实现（彩色图）：
```cpp
static inline void process_image_rvv(
    const uint8_t* img_data,
    int8_t* output,
    int img_h, int img_w,
    const float mean[3],
    const float scale[3]
) {
    size_t total_pixels = img_h * img_w;
    const uint8_t* p = img_data;
    int8_t* ptr_ch0 = output;
    int8_t* ptr_ch1 = ptr_ch0 + total_pixels;
    int8_t* ptr_ch2 = ptr_ch1 + total_pixels;

    size_t vl = vsetvlmax_e8m2();  // 最大向量长度（uint8_t, LMUL=2）
    vuint16m4_t v_zero = vmv_v_x_u16m4(0, vl);

    for (size_t n = total_pixels; n > 0; ) {
        if (vl > n) vl = n;
        n -= vl;

        // Step 1: Load RGB channels (HWC format, stride=3)
        vuint8m2_t v_r_u8 = vlse8_v_u8m2(p,     3, vl);
        vuint8m2_t v_g_u8 = vlse8_v_u8m2(p + 1, 3, vl);
        vuint8m2_t v_b_u8 = vlse8_v_u8m2(p + 2, 3, vl);

        // Step 2: u8 → u16 → u32 → f32
        vuint16m4_t v_r_u16 = vwcvtu_x_x_v_u16m4(v_r_u8, vl);
        vuint32m8_t v_r_u32 = vwcvtu_x_x_v_u32m8(v_r_u16, vl);
        vfloat32m8_t v_r_f32 = vfcvt_f_xu_v_f32m8(v_r_u32, vl);

        vuint16m4_t v_g_u16 = vwcvtu_x_x_v_u16m4(v_g_u8, vl);
        vuint32m8_t v_g_u32 = vwcvtu_x_x_v_u32m8(v_g_u16, vl);
        vfloat32m8_t v_g_f32 = vfcvt_f_xu_v_f32m8(v_g_u32, vl);

        vuint16m4_t v_b_u16 = vwcvtu_x_x_v_u16m4(v_b_u8, vl);
        vuint32m8_t v_b_u32 = vwcvtu_x_x_v_u32m8(v_b_u16, vl);
        vfloat32m8_t v_b_f32 = vfcvt_f_xu_v_f32m8(v_b_u32, vl);

        // Step 3: Apply (x - mean) * scale
        v_r_f32 = vfmul_vf_f32m8(vfsub_vf_f32m8(v_r_f32, mean[0], vl), scale[0], vl);
        v_g_f32 = vfmul_vf_f32m8(vfsub_vf_f32m8(v_g_f32, mean[1], vl), scale[1], vl);
        v_b_f32 = vfmul_vf_f32m8(vfsub_vf_f32m8(v_b_f32, mean[2], vl), scale[2], vl);

        // Step 4: f32 → i32 → i16 → i8 (with saturation)
        vint32m8_t v_r_i32 = vfcvt_x_f_v_i32m8(v_r_f32, vl);
        vint16m4_t v_r_i16 = vnclip_wv_i16m4(v_r_i32, v_zero, vl);
        vint8m2_t  v_r_i8  = vnclip_wx_i8m2(v_r_i16, 0, vl);

        vint32m8_t v_g_i32 = vfcvt_x_f_v_i32m8(v_g_f32, vl);
        vint16m4_t v_g_i16 = vnclip_wv_i16m4(v_g_i32, v_zero, vl);
        vint8m2_t  v_g_i8  = vnclip_wx_i8m2(v_g_i16, 0, vl);

        vint32m8_t v_b_i32 = vfcvt_x_f_v_i32m8(v_b_f32, vl);
        vint16m4_t v_b_i16 = vnclip_wv_i16m4(v_b_i32, v_zero, vl);
        vint8m2_t  v_b_i8  = vnclip_wx_i8m2(v_b_i16, 0, vl);

        // Step 5: Store in CHW format
        vse8_v_i8m2(ptr_ch0, v_r_i8, vl);
        vse8_v_i8m2(ptr_ch1, v_g_i8, vl);
        vse8_v_i8m2(ptr_ch2, v_b_i8, vl);

        // Step 6: Advance pointers
        p += vl * 3;
        ptr_ch0 += vl;
        ptr_ch1 += vl;
        ptr_ch2 += vl;
    }
}
```

### 灰度图简化版：
```cpp
static inline void process_image_gray_rvv(
    const uint8_t* img_data,
    int8_t* output,
    int img_h, int img_w,
    const float& mean,
    const float& scale
) {
    size_t total_pixels = img_h * img_w;
    const uint8_t* p = img_data;
    int8_t* ptr_ch0 = output;
    size_t vl = vsetvlmax_e8m2();
    vuint16m4_t v_zero = vmv_v_x_u16m4(0, vl);

    for (size_t n = total_pixels; n > 0; ) {
        if (vl > n) vl = n;
        n -= vl;

        vuint8m2_t v_r_u8 = vle8_v_u8m2(p, vl);
        vuint16m4_t v_r_u16 = vwcvtu_x_x_v_u16m4(v_r_u8, vl);
        vuint32m8_t v_r_u32 = vwcvtu_x_x_v_u32m8(v_r_u16, vl);
        vfloat32m8_t v_r_f32 = vfcvt_f_xu_v_f32m8(v_r_u32, vl);

        v_r_f32 = vfmul_vf_f32m8(vfsub_vf_f32m8(v_r_f32, mean, vl), scale, vl);

        vint32m8_t v_r_i32 = vfcvt_x_f_v_i32m8(v_r_f32, vl);
        vint16m4_t v_r_i16 = vnclip_wv_i16m4(v_r_i32, v_zero, vl);
        vint8m2_t v_r_i8 = vnclip_wx_i8m2(v_r_i16, 0, vl);

        vse8_v_i8m2(ptr_ch0, v_r_i8, vl);

        p += vl;
        ptr_ch0 += vl;
    }
}
```

> 💡 **提示**：上述代码将输入 `uint8_t` 图像归一化后转为 `int8_t` 输出（常见于量化模型输入）。可根据实际需求调整数据类型和缩放逻辑。

---

> ✅ **勘误或讨论？**  
> [查看已有 issue](https://github.com/neucrack/web/issues?q=%E3%80%90551%E3%80%91) 或 [提交勘误/讨论](https://github.com/neucrack/web/issues/new?assignees=&labels=article_err&template=article_error.md&title=【551】【勘误】 我是标题,修改我,拷贝链接到下方 https://neucrack.com/p/551)（需 GitHub 登录）