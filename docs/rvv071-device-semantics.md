# SG2002/C906 RVV 0.7.1 真机语义差异（实测）

日期：2026-11-20
设备：MaixCAM（192.168.2.216），Buildroot，Python 3.11.6，`isa: rv64imafdvcsu`
工具链：T-Head Xuantie-900 musl GCC 10.2.0（V2.6.1 B-20220906）
flags：`-mcpu=c906fdv -march=rv64imafdcv0p7xthead -mcmodel=medany -mabi=lp64d`

## 为什么需要这份记录

`rvv` 0.2.1 在 x86 标量路径上全部测试通过，交叉编译也成功（ELF 里有 RVV 指令），
但**真机一跑就大面积出错**。逐项探针（`probe/rvv_probe.cpp`、`probe/rvv_compare_probe.cpp`）
实测出以下差异，都必须在写 RVV 代码时规避。

## 1. VLMAX 与 tail-agnostic

实测：

```text
e8m1=16  e8m4=64  e8m8=128
e16m2=16 e16m4=32
e32m1=4  e32m2=8  e32m4=16 e32m8=32
```

`vsetvlmax_e32m4() = 16`，即 128-bit VLEN / 32-bit element × LMUL 4。

**tail lane 是 agnostic，会被写坏**：

```c
vfloat32m2_t acc = vfmv_v_f_f32m2(1.0f, vlmax);   // 全 1
acc = vfadd_vv_f32m2(acc, addend, 1);             // vl=1
// 实测 acc lanes: 101 0 0 0 0 0 0 0
// 其它 lane 变成 0，不是保持 1
```

⇒ **跨迭代累加绝不能混用不同 vl**。一旦某次 `vl < vlmax`，累加器里超过 `vl`
的 lane 就被写成未定义值，后续再做归约就会把垃圾算进去。

实测后果：

```text
20 个 1.0，用「vl=vsetvl(n-i) 增量累加 + 按 vlmax 归约」  => 12.0  ❌
20 个 1.0，用「整 vlmax 块 + 标量尾部」                    => 20.0  ✅
```

**规则**：跨迭代的向量累加器只处理整 `vlmax` 块；剩余元素走标量。

## 2. 水平归约的 vl 必须是 vlmax

原 `dot()` 写成：

```c
vfloat32m4_t vsum = vfmv_v_f_f32m4(0.0f, 4);            // 4，不是 vlmax
vred = vfredosum_vs_f32m4_f32m1(vred, vsum, vred, 4);    // 4，不是 vlmax
```

手册原文（p.29《Configuration-Setting and vl Argument》）：

> The intrinsic functions will only operate at most VLMAX element if the vl
> argument are larger than VLMAX.

反过来同样成立：**vl 小于 VLMAX 时就只算 vl 个**。所以归约只加了前 4 个 lane。

实测：

```text
n=4    dot=4      (expect 4)     ✅ 巧合正确
n=8    dot=4      (expect 8)     ❌
n=16   dot=4      (expect 16)    ❌
n=1000 dot=252    (expect 1000)  ❌
```

**规则**：`vfredsum/vfredusum/vfredmax/vfredmin/vredsum` 的 `vl` 用 `vsetvlmax_*()`。

## 3. `vmf*_vf`（标量比较）语义错误 —— 最危险的坑

真机实测（`vals = {100, 250, 255, 249, 250.5, 0, 251, 240}`，阈值为 250）：

| intrinsic | 实测结果 | 正确结果 |
|---|---|---|
| `vmfge_vf(c, 250)` | `1 0 0 1 0 1 0 1` | `0 1 1 0 1 0 1 0` |
| `vmfgt_vf(c, 250)` | `1 1 0 1 0 1 0 1` | `0 1 1 0 1 0 1 0` |
| `vmfle_vf(c, 250)` | `0 0 1 0 1 0 1 0` | `1 1 0 1 1 0 1 1` |
| `vmflt_vf(c, 250)` | `0 1 1 0 1 0 1 0` | `1 0 0 1 0 1 0 1` |

`vmfne_vf(c, 250)` 也只在部分 lane 上正确（`0 1 0 0 0 0 0 0`）。

对比 `vfmax_vf(c, 250)` 是正确的：`250 250 255 250 250.5 250 251 250`。

⇒ **这套工具链的 `vmf*_vf` 全部不可用。** 需要用比较时，改成
`vmf*_vv`（向量对向量）或直接在标量侧处理。

## 4. `vfirst_m_b16` 返回值不可信

```text
vfirst_m_b16(vmfgt_vf(c,250)) = 2   (expect 1)
vfirst_m_b16(vmflt_vf(c,250)) = 0   (expect 0)   ← 巧合
vfirst_m_b16(vmfne) = 16            (mask 只有 8 lane，返回 16 越界)
```

手册 p.24 提到 `vpopc.m` / `vfirst.m` 在 0.7.1 与 1.0 之间有改名，且返回类型是整数。
实测值与手册描述不一致 ⇒ **不要用 `vfirst_m` 做任何判断**。

## 5. `vmfne_vv` 做 NaN 检测不可用

```text
withnan = {1, 2, NaN, 4, 5, 6, 7, 8}
vmfne(vn, vn) bits = 1 1 0 1 1 1 1 1   (expect 0 0 1 0 0 0 0 0)
vfirst_m_b16(vmfne) = 16
```

⇒ `max_f32` / `min_f32` 原来的 NaN 检测恒为真，导致**任何输入都返回 NaN**。
NaN 检测改为标量侧 `std::isnan`。

## 6. 可以放心使用的部分（实测正确）

- `vle32/vse32/vle8/vse8/vlse8`（含 `vlse8` stride=3 的 RGB 反交错，结果正确）
- `vwaddu_vx_u16m2` + `vfwcvt_f_xu_v_f32m4` 的 u8→u16→f32 拓宽（值正确）
- `vfadd/vfsub/vfmul/vfmax/vfmin` 的 vv/vf 形式
- `vmerge_vvm`（mask 为真取第一个操作数，实测一致）
- `vfredmax/vfredmin`、`vredsum`（vl 用 vlmax 时正确）
- `vfredusum`（vl 用 vlmax 时正确）

## 7. 由此确定的编码规则

1. **跨迭代向量累加器只吃整 vlmax 块**，尾部用标量。
2. **所有归约的 vl 用 `vsetvlmax_*()`**，永不用常数或 `vsetvl` 的返回值。
3. **禁用 `vmf*_vf`**；需要比较时用 `vmf*_vv` 或标量。
4. **禁用 `vfirst_m`、`vmfne_vv` 做判断**；NaN/计数在标量侧做。
5. mask + `vmerge_vvm_u32m2(mask, one, zero, vl)` 可以安全用于计数累加，
   但累加器同样要遵守规则 1。
6. 每加一个 RVV 函数，**必须真机验证**；x86 标量测试和交叉编译都不能发现这些错误。

## 8. 复现方式

```bash
# 主机（容器 rvv-build）
riscv64-unknown-linux-musl-g++ -O2 -std=c++17 -static \
  -march=rv64imafdcv0p7xthead -mcpu=c906fdv -mcmodel=medany -mabi=lp64d \
  probe/rvv_probe.cpp -o rvv_probe
scp rvv_probe root@192.168.2.216:/tmp/
ssh root@192.168.2.216 chmod +x /tmp/rvv_probe '&&' /tmp/rvv_probe
```

探针源码：`D:\projects\rvv-build\probe\rvv_probe.cpp`、
`D:\projects\rvv-build\probe\rvv_compare_probe.cpp`。
