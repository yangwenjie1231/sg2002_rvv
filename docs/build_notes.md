# 构建注意事项与真机验证要求

目标：SG2002 / C906FDV，RVV 0.7.1，CPython 3.11.6，musl，riscv64，`lp64d`。

## 1. 工具链与 flags

- 工具链：T-Head Xuantie-900 musl GCC 10.2.0
- flags：`-mcpu=c906fdv -march=rv64imafdcv0p7xthead -mcmodel=medany -mabi=lp64d`
- 完整获取方式与 SHA256 见 `BUILD_ENV.md`

**不要加 `-mrvv-vector-bits=128`**：T-Head GCC 10.2 的 `riscv_vector.h` 会报
`invalid argument to built-in function`。

## 2. 扩展文件名

目标 Python 3.11.6 虽然运行在 musl 上，但 `_imp.extension_suffixes()` 实测为：

```text
['.cpython-311-riscv64-linux-gnu.so', '.abi3.so', '.so']
```

⇒ `.so` 必须命名为 `rvv.cpython-311-riscv64-linux-gnu.so`，**不能**用
`...-linux-musl.so`，否则设备 `import rvv` 找不到模块。

wheel 的平台标签仍可用 `musllinux_1_2_riscv64` 描述 libc，两者不冲突。

## 3. 链接

建议 `-static-libstdc++ -static-libgcc`，产物只依赖 `libc.so`，避免设备缺
`libstdc++.so.6`。

## 4. 真机验证是强制的

0.2.1 的教训：全部 x86 标量测试通过、交叉编译成功、ELF 里也有 RVV 指令，
**但真机上 dot/norm_l2/sum/max/min 全部算错**，`gradient_stats_u8` 的
`grad_count`/`saturated_count` 也全错。原因是真机的 intrinsic 语义与预期不同：

- tail lane 是 **agnostic**（`vl < vlmax` 时越界 lane 变未定义值）；
- `vmf*_vf`（标量比较）语义错误；
- `vfirst_m` 返回值不可信。

详见 `docs/rvv071-device-semantics.md`。

⇒ 任何 RVV 代码改动都必须：

1. 在设备上跑算子回归（与纯 Python 参考实现逐项对比）；
2. 覆盖 `n` 为 1、3、4、7、8、9、15、16、17、33、100、1000 等边界；
3. 覆盖 `gradient_stats_u8` 的多种 `w×h`、`roi_fraction`、`step`、`offset`、灰度/RGB。

x86 标量测试只能证明算法逻辑，不能证明 RVV 路径。

## 5. wheel 架构验证

```bash
unzip -o rvv-*.whl -d /tmp/rvv-wheel
SO=$(find /tmp/rvv-wheel -name '*.so' | head -1)
file "$SO"                                   # ELF 64-bit RISC-V
readelf -A "$SO" | grep Tag_RISCV_arch       # 含 v0p7 ... xtheadc2p0
nm -D "$SO" | grep -E 'PyInit|gradient_stats'
```

不要用 `py3-none-any` 打运行库（会「安装成功、运行失败」）；stub-only 包才用
`py3-none-any`，且只放 `.pyi`。

## 6. 当前产物

```text
rvv-0.2.2-cp311-cp311-musllinux_1_2_riscv64.whl
sha256: e8274443ffecc00d9621b9f28bfeab72ce61c57a583450d31ff9ae9cb207e424
```

真机（MaixCAM，192.168.2.216）回归：**223/223 通过**；
`gradient_stats_u8` 640×480 RGB **3.5 ms/次**。

## 7. 性能说明（避免误判）

真机上 `rvv.add(list, list)` 比纯 Python 列表推导还慢，原因是 Python list →
`float32` 的转换成本（实测 100k 元素：list 476 ms、`array('f')` 12 ms、
`numpy` 8.9 ms）。

⇒ **不要用 Python list 做性能对比**；固件场景应传 `bytes`/`memoryview`
（`gradient_stats_u8` 走 buffer 路径，无转换开销）。
