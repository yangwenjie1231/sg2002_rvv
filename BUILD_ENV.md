# SG2002 / C906 RVV 0.7.1 构建环境

本文记录 `rvv` 的可复现交叉构建方式。目标：

- SoC：SOPHGO SG2002 / T-Head C906FDV
- ISA：RVV 0.7.1（128-bit）
- Python：3.11.6，musl，riscv64，`lp64d`
- 扩展：`rvv.cpython-311-riscv64-linux-gnu.so`

> 本仓库不再提交 `host-tools`、SDK、`build/`、`dist/` 或 wheel 二进制。
> 工具链请从 MaixCDK 的 `platforms/maixcam.yaml` 获取。

## 1. 工具链

来源：MaixCDK `platforms/maixcam.yaml`

```text
https://github.com/sipeed/MaixCDK/releases/download/v0.0.0/host-tools-2025.7.28.tar.xz
sha256: 968b179a7faebfe8914e5163b2d85038b71824780fccde0b4b3ee9f67472d007
```

解压后编译器：

```text
host-tools/gcc/riscv64-linux-musl-x86_64/bin/riscv64-unknown-linux-musl-g++
```

目标 Python 头文件/库来自 MaixCDK release：

```text
python3_lib_maixcam_musl_3.11.6.tar.xz
sha256: 92e14b22e708612c6e280931cc247b4266da9a6bac8459edf25bfb4cebcbac66
```

## 2. Docker 构建

```bash
docker run --rm -it \
  -v "$PWD":/work \
  -v /path/to/toolchain:/opt/maixcam \
  -v /path/to/python3_lib_maixcam_musl_3.11.6:/opt/py311 \
  ubuntu:24.04 bash
```

容器内安装 `build-essential cmake ninja-build pybind11` 后，执行：

```bash
export PATH=/opt/maixcam/host-tools/gcc/riscv64-linux-musl-x86_64/bin:$PATH
riscv64-unknown-linux-musl-g++ \
  -O3 -std=c++17 -fPIC -shared \
  -march=rv64imafdcv0p7xthead -mcpu=c906fdv \
  -mcmodel=medany -mabi=lp64d \
  -I/opt/py311/include/python3.11 \
  -I/usr/include/pybind11 \
  -Isrc src/*.cpp \
  -o rvv.cpython-311-riscv64-linux-gnu.so
```

注意：

- **不要**加 `-mrvv-vector-bits=128`。T-Head GCC 10.2 的 `riscv_vector.h` 会报
  `invalid argument to built-in function`。
- 目标 Python 的 `EXT_SUFFIX` 是 `rvv.cpython-311-riscv64-linux-gnu.so`，即使
  libc 是 musl；不要命名成 `...-linux-musl.so`。
- 链接建议使用 `-static-libstdc++ -static-libgcc`，避免设备缺 `libstdc++.so.6`。

## 3. 验证

```bash
file rvv.cpython-311-riscv64-linux-gnu.so
readelf -A rvv.cpython-311-riscv64-linux-gnu.so | grep Tag_RISCV_arch
nm -D rvv.cpython-311-riscv64-linux-gnu.so | grep -E 'PyInit|mul_f32|max_f32'
```

预期：

```text
ELF 64-bit LSB shared object, UCB RISC-V
Tag_RISCV_arch: rv64i2p0...v0p7...xtheadc2p0
```

标量路径可在 x86 主机用同一份 `src/*.cpp` 编译并运行 `tests/test_*.py`；RVV 指令
路径需要在真实 SG2002 / MaixCAM 上验证，QEMU 的 `thead-c906` 不能可靠执行完整模块。

## 4. 发布 wheel

当前 preview wheel：

```text
rvv-0.2.1-cp311-cp311-musllinux_1_2_riscv64.whl
sha256: d23e853393d10163197460fd9bd8ab79b520959c3de89f40ede89b3d3b9c1600
```

如果目标 pip 不接受 `musllinux_1_2_riscv64` 标签，可使用同一 .so 打包的
`linux_riscv64` 兼容 wheel。
