# RVV 扩展构建与 ABI 说明（SG2002 / C906 / RVV 0.7.1）

状态：**调研与构建约束说明**。本文不执行构建、不修改 `src/`，只记录可复现的工具链、flags、ABI 与验证方法。

目标：产出能在 MaixCAM（SG2002，C906FDV，RVV 0.7.1，VLEN=128）上被 **CPython 3.11.6 (musl, riscv64, lp64d)** 导入的 `rvv` 扩展 wheel；同一份源码在 x86 主机上应能走标量路径做单元测试。

---

## 1. 目标平台事实

| 项目 | 值 | 依据 |
|---|---|---|
| SoC | SOPHGO SG2002 | Sipeed MaixCAM 硬件页 |
| 大核 | 1 GHz RISC-V C906（Linux） | 同上 |
| 向量扩展 | RVV **0.7.1-Workshop-Release** | T-Head C906 规格；intrinsic 手册 Preface |
| VLEN | **128 bit** | T-Head C906 规格；RISC-V International 介绍 |
| 元素宽度 | 8/16/32/64 bit | T-Head C906 规格 |
| ABI | `lp64d`（64-bit long/pointer，hard-float double） | MaixCDK `platforms/maixcam.yaml` |
| libc | musl | 同上 |
| Python | **3.11.6**，SDK 组件 `python3_lib_maixcam_musl_3.11.6` | `sg2002_rvv` README；MaixPy `docs/doc/zh/pro/rvv.md` |

关键结论：这是 **RVV 0.7.1**，不是 RVV 1.0。`-march=...v` 或 `-march=rv64gcv` 的泛化写法不能用于本项目；必须带 `v0p7` 版本后缀。

参考：

- Sipeed MaixCAM 硬件页：<https://wiki.sipeed.com/hardware/en/maixcam/maixcam.html>（CPU: SG2002，1 GHz RISC-V C906）。
- T-HEAD C906 规格：<https://www.riscvschool.com/2023/03/09/t-head-xuantie-c906-risc-v/>（RVV 0.7.1、Vector length 128 bits）。
- RISC-V International C906 介绍：<https://riscv.org/blog/xuantie-c906-tops-mlperf-tiny-v0-7-benchmark-mengchang-alibaba-cloud/>（128-bit vector operation units）。
- 本地手册：`D:\projects\rvv-ext\Xuantie+900+Series+RVV-0.7.1+Intrinsic+Manual.pdf`，Preface p.1。

---

## 2. 工具链

### 2.1 推荐来源（MaixCDK 固化版本）

MaixCDK 的 MaixCAM 平台配置给出了可直接复现的工具链：

- 文件：<https://github.com/sipeed/MaixCDK/blob/main/platforms/maixcam.yaml>
- 下载：<https://github.com/sipeed/MaixCDK/releases/download/v0.0.0/host-tools-2025.7.28.tar.xz>
- SHA-256：`968b179a7faebfe8914e5163b2d85038b71824780fccde0b4b3ee9f67472d007`
- 旧版备用：<https://sophon-file.sophon.cn/sophon-prod-s3/drive/23/03/07/16/host-tools.tar.gz>
- 解压后的编译器目录：`toolchains/maixcam/host-tools/gcc/riscv64-linux-musl-x86_64/bin`
- 编译器前缀：`riscv64-unknown-linux-musl-`

> 用绝对路径配置 `CONFIG_TOOLCHAIN_PATH` / `TOOLCHAIN_PATH`；上游 README 明确记录相对路径会报 `TOOLCHAIN_PATH set error`。

### 2.2 规范 flags

MaixCDK `maixcam.yaml` 的 `c_flags` / `cxx_flags` 是：

```text
-mcpu=c906fdv
-march=rv64imafdcv0p7xthead
-mcmodel=medany
-mabi=lp64d
```

建议的 C++ 编译命令（仅作示例，本文不执行）：

```bash
TOOLCHAIN=/path/to/toolchains/maixcam/host-tools/gcc/riscv64-linux-musl-x86_64/bin
CXX="$TOOLCHAIN/riscv64-unknown-linux-musl-g++"

"$CXX" -std=c++17 -O3 -fPIC -shared \
  -mcpu=c906fdv \
  -march=rv64imafdcv0p7xthead \
  -mcmodel=medany \
  -mabi=lp64d \
  -I./src -I"$(python3.11 -c 'import pybind11; print(pybind11.get_include())')" \
  src/*.cpp -o rvv.so
```

（实际 pybind11 构建还应带上 Python 3.11 的 include 路径；这里只突出目标 flags。）

补充说明：

- `-O3` 是性能构建的合理默认；`-Os` 可用于减小体积，但不建议牺牲向量循环性能。
- `-mrvv-vector-bits=128` 出现在本仓库的 `compile_flags.txt` / `sdkconfig` 中，但 **不在** MaixCDK 的 `maixcam.yaml` 规范 flags 内。它不是所有 T-Head GCC 版本都支持；使用前用 `$CXX --help=target` 或最小编译验证。C906 的 VLEN 固定 128，通常不需要该 flag。
- `-flto`：本仓库 `CMakeLists.txt` 明确过滤掉 `-flto`；不要自行加，除非完整验证过交叉链接和 pybind11 导出符号。
- `-mcmodel=medany`、`-mabi=lp64d` 必须与目标 rootfs 中所有库一致；不要混用 `lp64` / `lp64f`。

### 2.3 本仓库当前 flags 的风险

当前 `sdkconfig` / `parse_sdkconfig.py` 会生成重复且可能互相覆盖的 `-march`：

```text
-march=rv64gcv0p7
-mcpu=c906fdv
-march=rv64imafdcv0p7xthead
```

GCC 通常以最后一个 `-march` 为准，所以结果可能正确，但这是**脆弱配置**：一旦顺序变化就会退回通用 `rv64gcv0p7`（缺少 `xthead`，甚至可能被新版工具链解释成不同版本）。建议只保留 MaixCDK 的四个规范 flags，不要在 `-mcpu` 前后再插一个泛化 `-march`。

另一个确定的问题：`tools/parse_sdkconfig.py` 只写 `compile_flags.txt`，对 `compile_definitions.txt` 写的是空 `defs` 列表（第 26–28 行）。因此执行 `python project.py build` 会用空文件覆盖当前仓库里已有的：

```text
RVV_PROJECT_VERSION=1.0.0
RVV_BUILD_RELEASE
__RISCV_VECTOR_VERSION=0x00070100
__riscv_vector
```

其中 `__riscv_vector` 是编译器保留宏，**不应该手工定义**；`__RISCV_VECTOR_VERSION` 也不是可靠的标准版本判据。修复构建脚本时应让编译器自己定义 `__riscv_vector`，并用构建系统传入项目自己的版本字符串（例如 `RVV_PROJECT_VERSION`）。

---

## 3. Python 3.11.6 / musl / ABI

### 3.1 目标解释器

- 目标：CPython **3.11.6**，musl，riscv64，`lp64d`。
- SDK 组件名：`python3_lib_maixcam_musl_3.11.6`（`sg2002_rvv` README；MaixPy `docs/doc/zh/pro/rvv.md` §“构建模式选择”）。
- MaixPy 自己的打包配置也把 MaixCAM 定为 Python `[3, 11]`，平台 `linux_riscv64`，工具链 `musl_t-head`（`MaixPy/setup.py`）。

### 3.2 wheel tag

正确形态应类似：

```text
rvv-0.1.3-cp311-cp311-linux_riscv64.whl
```

要点：

- **Python tag 必须是 `cp311`**，不是 `py3`、`cp313`。补丁版本 3.11.6 不进入 ABI tag；CPython 的稳定 ABI 以 `cp311` 表示。
- **平台 tag 必须是 `linux_riscv64`**（与 MaixPy 的 `--plat-name linux_riscv64` 一致），不是 `any`，也不是 `manylinux*`。
- 不应使用 `py3-none-any`：本扩展含原生 `.so`，`any` 会让 x86 主机也尝试安装，随后在导入时失败。
- 不建议用 `abi3`：pybind11 默认不使用 Limited API；若要 `abi3`，需要单独验证且会限制可用 API。

### 3.3 构建主机

- 必须用 **Python 3.11** 的解释器和头文件构建；当前开发机默认 Python 3.13.5 不能直接产出 `cp311` 扩展（即使代码能编译，扩展后缀和 ABI 也不匹配）。
- `setup.py` 目前写 `python_requires=">=3.8"`，对 MaixCAM 不准确；应改为 `>=3.11,<3.12`（或 `==3.11.*`），并由 wheel tag 再次约束。
- `setup.py` 目前写 `install_requires=["numpy"]`，但设备端通常无法安装/编译 NumPy；这与“wheel 安装不依赖 NumPy”的目标冲突。若 API 设计为接受 buffer、返回 dict，应改为 `install_requires=[]`，NumPy 只作为可选测试依赖。

### 3.4 C++ ABI / 动态库

- 使用同一套 T-Head musl 工具链编译所有 C++ 源；不要用主机 glibc 的 `g++` 链接目标 `.so`。
- 检查 `readelf -d` 的 `NEEDED`：应看到 `libc.musl-riscv64.so.1`、`libstdc++.so.6`、`libgcc_s.so.1` 等目标库；不能出现主机 glibc 路径。
- 若目标 rootfs 的 `libstdc++` 较旧，可考虑 `-static-libstdc++ -static-libgcc`，但会增加体积并需验证异常/线程行为；默认先动态链接，按真机 `readelf`/`ldd` 结果决定。
- `-fvisibility=hidden` 可用；pybind11 的 `PYBIND11_MODULE` 会正确导出 `PyInit_rvv`，但需确认没有因隐藏符号导致导入失败。

---

## 4. `__riscv_vector` 检测

### 4.1 它是什么

GCC 在启用 RISC-V V 扩展时定义 `__riscv_vector`。本项目用它选择 RVV 实现：

```cpp
#if defined(__riscv_vector)
    // RVV 0.7.1 intrinsics
#else
    // scalar fallback
#endif
```

手册 Preface（p.1）说明：使用 RVV 0.7.1 时 arch 选项要写 `v0p7`，而不是泛化的 `v`。

### 4.2 它不能区分 0.7.1 和 1.0

`__riscv_vector` 在 RVV 0.7.1 和 RVV 1.0 工具链上都可能被定义。因此：

- **不能**用 `#ifdef __riscv_vector` 证明“这是 0.7.1”；
- 必须同时固定工具链和 `-march=...v0p7...`；
- 若工具链定义了 `__riscv_v_intrinsic`，可以打印/记录其值作为辅助证据；未定义并不等于 0.7.1；
- 不要手工 `-D__riscv_vector`。这会让标量主机错误编译 RVV 分支，而且该宏属于编译器实现保留标识。

### 4.3 建议的构建期检查

编译前检查预定义宏：

```bash
"$CXX" -march=rv64imafdcv0p7xthead -mabi=lp64d -dM -E -x c++ /dev/null \
  | grep -E '__riscv_vector|__riscv_v_intrinsic|__riscv_arch|__riscv_xlen'
```

期望至少看到 `#define __riscv_vector 1`；若没有，说明 flags 没生效，RVV 源文件会走标量分支。

对 RVV 实现文件可加编译期保护：

```cpp
#if !defined(__riscv_vector)
#error "RVV path selected but __riscv_vector is undefined; check -march=...v0p7..."
#endif
```

该保护只放在 `rvv_rvv071.cpp` 这类 RVV-only 文件里，不要放进标量 fallback 文件，否则 x86 主机测试无法编译。

### 4.4 目标文件检查

编译后检查是否真的生成了向量指令：

```bash
"$TOOLCHAIN/riscv64-unknown-linux-musl-objdump" -d build/rvv.so \
  | grep -E '\bvsetvli\b|\bvlse8\b|\bvwmulu\b|\bvredsum\b' \
  | head
```

若只有标量循环、没有 `vsetvli`，优先怀疑：

1. `__riscv_vector` 未定义（flags 未生效）；
2. 源文件被错误地排除；
3. 编译器内联/优化把循环矢量化成了标量（查看 `-fopt-info-vec`）；
4. 链接了错误目标文件。

---

## 5. RVV 0.7.1 vs RVV 1.0：不要混用

手册 Preface（p.1）明确：

> 使用 RVV 0.7.1 时，arch 选项使用 `v0p7`，不要使用 `v`。

手册第 20 章（p.19–24）列出 0.7.1 intrinsic 与 1.0 的差异。以下是与本项目最相关的几类（左边是 0.7.1，右边是 1.0）：

| 类别 | RVV 0.7.1 | RVV 1.0 | 影响 |
|---|---|---|---|
| 无序 indexed load/store | `vlxei<eew>.v` / `vsxei<eew>.v` | `vloxei` / `vsoxei` | 直接换名；照抄 1.0 名会找不到 intrinsic |
| 带扩展的 load/store | `vlb/vlh/vlw`、`vlsb/...` 等 | 删除，改用 `vzext/vsext` + `vle` | 图像 u8→u16 的旧写法在 1.0 头文件不存在 |
| narrowing shift | `vnsra.vv` / `vnsrl.vv` | `vnsra.wv` / `vnsrl.wv` | 参数类型和寄存器组约束不同 |
| narrowing clip | `vnclipu.vx` / `vnclip.vx` | `vnclipu.wx` / `vnclip.wx` | 同上 |
| mask 操作 | `vmpopc.m` / `vmfirst.m` / `vmcpy.m` | `vpopc.m` / `vfirst.m` / `vmmv.m` | 计数器/首个置位实现不同 |
| 浮点 narrowing convert | `vfncvt.x.f.v` 等 | `vfncvt.x.f.w` 等 | 参数宽度不同 |
| 点积 | 第 18 章 `vdot` intrinsic 列表仍为 TODO | 1.0 有独立 dot 支持 | **不要依赖 `vdot`**；用 widening multiply + reduction |

本项目的具体实现建议：

- 只用手册中明确列出的 0.7.1 intrinsic（如 `vlse8_v_u8m*`、`vwmulu`、`vwmaccu`、`vredsum`、`vwredsumu`）。
- 不把 `vdot` 作为性能前提；手册 p.19 的 `vdotu/vdot` 章节 intrinsic 仍标 TODO。
- 不把 0.7.1 的 `.so` 发布成“通用 RISC-V wheel”。`linux_riscv64` tag 不区分 RVV 版本；在 RVV 1.0 板子上可能直接非法指令或链接失败。至少要：
  - 在包名/描述/README 中明确 `C906 / RVV 0.7.1 only`；
  - 在模块初始化时暴露编译期 arch 字符串（例如 `rvv.__rvv_arch__ = "rv64imafdcv0p7xthead"`），便于现场排查；
  - 若未来要支持 RVV 1.0，单独构建、单独 wheel、单独 CI，不共享同一个二进制。

### 5.1 为什么不能直接用 `-march=rv64gcv`

- `rv64gcv` 通常表示已批准的 V 扩展（1.0 语义），而不是 0.7.1；
- 即使编译器暂时接受，intrinsic 头文件、指令名、mask 类型、narrowing 规则都可能不同；
- C906 的 GCC 需要 `-march=rv64imafdcv0p7xthead`（MaixCDK 固化值），其中 `xthead` 是 T-Head 自定义扩展，不能省略。

### 5.2 运行期不要尝试“自动兼容 1.0”

RVV 0.7.1 与 1.0 的差异是**编译期 intrinsic/指令编码差异**，不是运行期可以简单分支的 API 差异。一个二进制无法同时安全覆盖两者；正确做法是分架构构建。

---

## 6. 当前仓库的构建链风险（需修，但本次不改 `src/`）

### 6.1 CMake 与 setuptools 各自编译一次

当前 `CMakeLists.txt`：

- 用 `pybind11_add_module(rvv ${SOURCES})` 生成 `rvv`；
- `install(TARGETS rvv DESTINATION build/dist)`；
- 构建后执行 `python3 setup.py bdist_wheel`。

但 `setup.py` 又定义了一个独立的 `Extension("rvv", src/*.cpp, extra_compile_args=[...])`，因此 `bdist_wheel` 会**重新编译**一份，而不是打包 CMake 生成的 `.so`。这会导致：

- 两份二进制的 flags/ABI 可能不同；
- 交叉编译时，第二次编译若没有继承 `CC/CXX`/`CFLAGS`，会用主机编译器；
- 修复时应二选一：
  1. **CMake 主导**：CMake 生成 `.so`，打包脚本只复制并用 `Distribution.has_ext_modules=True` + `--plat-name linux_riscv64 --python-tag cp311` 打 wheel；
  2. **setuptools 主导**：删除 CMake 的二次打包，显式设置交叉编译器/环境，确保 `Extension` 使用 T-Head 工具链。

### 6.2 `python_requires` / `install_requires`

- `python_requires=">=3.8"` 与 `cp311` 目标不一致；改为 `>=3.11,<3.12`。
- `install_requires=["numpy"]` 会让设备端 pip 尝试安装 NumPy；改为 `[]`，把 NumPy 放在 `extras_require["test"]`。

### 6.3 `compile_definitions.txt` 被清空

见 §2.3。不要依赖当前文件里的 `__riscv_vector` / `__RISCV_VECTOR_VERSION`；它们会在 `project.py build` 时被覆盖。

### 6.4 重复的 `-march`

见 §2.3。保留一套规范 flags，避免“最后一个 `-march` 赢”的隐式行为。

---

## 7. wheel 架构验证清单

以下命令在**构建机**执行，不需要目标板；最后两步在 MaixCAM 上执行。

### 7.1 文件名和 tag

```bash
ls -l dist/*.whl
python3 - <<'PY'
import glob, os, re
for p in glob.glob("dist/*.whl"):
    name = os.path.basename(p)
    print(name)
    assert re.search(r"-cp311-cp311-linux_riscv64\.whl$", name), \
        "wheel tag is not cp311-cp311-linux_riscv64"
PY
```

如果看到 `py3-none-any`，立即判定为**打包错误**：它会把 RISC-V `.so` 当成纯 Python 包发布。

### 7.2 包内扩展名

```bash
unzip -l dist/rvv-*.whl
```

应看到类似：

```text
rvv/rvv.cpython-311-riscv64-linux-musl.so
```

确切的 `EXT_SUFFIX` 取决于目标 Python 的构建配置；以目标机为准：

```bash
# 在 MaixCAM 上
python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
```

### 7.3 ELF 架构 / ISA 属性

```bash
rm -rf /tmp/rvv-wheel && mkdir -p /tmp/rvv-wheel
unzip -o dist/rvv-*.whl -d /tmp/rvv-wheel
SO=$(find /tmp/rvv-wheel -name '*.so' | head -1)

file "$SO"
readelf -h "$SO" | grep -E 'Class|Machine|Type'
readelf -A "$SO" | grep -E 'Tag_RISCV_arch|Tag_RISCV_priv_spec' || true
readelf -d "$SO" | grep NEEDED
```

期望：

- `file`：`ELF 64-bit LSB shared object, UCB RISC-V`；
- `readelf -h`：`Machine: RISC-V`；
- `readelf -A`：`Tag_RISCV_arch` 包含 `rv64imafdcv0p7xthead`（或至少 `...v0p7...`）；若显示 `v1p0` 或没有 V，说明 flags 错；
- `readelf -d`：`NEEDED` 包含 musl 的 `libc.musl-riscv64.so.1`，不应出现主机 glibc 路径。

### 7.4 符号

```bash
"$TOOLCHAIN/riscv64-unknown-linux-musl-nm" -D "$SO" | grep PyInit_rvv
```

必须存在 `PyInit_rvv`；否则 Python 导入会报 `dynamic module does not define module export function`。

### 7.5 向量指令

```bash
"$TOOLCHAIN/riscv64-unknown-linux-musl-objdump" -d "$SO" \
  | grep -E '\bvsetvli\b|\bvlse8\b|\bvwmulu\b' | head
```

没有向量指令时，先按 §4.3 检查预定义宏，再检查是否编译了 RVV 源文件。

### 7.6 目标板导入与标签

```bash
# 在 MaixCAM 上
python3 -c "import sys, sysconfig; print(sys.version); print(sysconfig.get_platform()); print(sysconfig.get_config_var('EXT_SUFFIX'))"
python3 -m pip debug --verbose | grep -E 'cp311|riscv'
python3 -c "import rvv; print(rvv.__file__)"
```

`pip debug` 的兼容 tag 列表应包含 `cp311-cp311-linux_riscv64`。若没有，`pip install` 会拒绝正确 wheel，或错误接受 `any` wheel。

### 7.7 负面验证（重要）

在 x86 Linux 主机上：

```bash
python3 -m pip install --no-deps dist/rvv-*.whl
```

- 正确 tag 的 wheel 应被 pip **拒绝**（不兼容平台）；
- `py3-none-any` 的 wheel 会被接受，随后 `import rvv` 才失败。这正是必须避免的“安装成功、运行失败”。

---

## 8. 建议的最小构建流程（目标状态）

1. 解压 MaixCDK 工具链，设置绝对 `TOOLCHAIN_PATH`。
2. 使用 Python 3.11 + pybind11，在项目根目录配置 CMake：
   ```bash
   cmake -B build -GNinja \
     -DCMAKE_TOOLCHAIN_FILE=tools/toolchain.cmake \
     -DPYTHON_EXECUTABLE=/path/to/python3.11
   ```
3. 只保留一套 flags：
   ```text
   -mcpu=c906fdv -march=rv64imafdcv0p7xthead -mcmodel=medany -mabi=lp64d
   ```
4. 构建并检查 `readelf -A` / `objdump`。
5. 用 `cp311` + `linux_riscv64` 打 wheel；不要用 `py3-none-any`。
6. 在 MaixCAM 上 `pip install`，导入并跑一个最小函数；再跑 `gradient_stats_u8` 的真机基准。
7. 在 x86 上用同一份源码构建标量版本，跑单元测试；不要试图在 x86 上运行 RISC-V `.so`。

---

## 9. 未确认项

1. **T-Head GCC 对 `-mrvv-vector-bits=128` 的支持**：本仓库使用了该 flag，但 MaixCDK 规范 flags 中没有；需用实际工具链验证，若不支持则删除。
2. **当前 `setup.py` 的交叉编译行为**：它没有显式设置 `CC/CXX`/`CFLAGS`，在不同环境下可能调用主机编译器；需在真实构建机复现并修正。
3. **目标 Python 的精确 `EXT_SUFFIX`**：`linux-riscv64` 与 `linux-musl` 的具体后缀依 CPython 构建配置而异；以目标机 `sysconfig` 输出为准。
4. **`__riscv_v_intrinsic` 在 T-Head 0.7.1 头文件中的定义情况**：只把它当辅助信息，不作为唯一版本判据。
5. **wheel 发布策略**：`linux_riscv64` 无法表达“仅 C906 / RVV 0.7.1”；是否拆成独立发行名（如 `rvv-c906`）需与维护者确认。
6. **真机 ABI 兼容性**：目标 rootfs 的 `libstdc++`/`libgcc_s` 版本需用 `readelf -d` + 实际导入验证；本文未在硬件上执行。

---

## 10. 参考

- MaixCDK MaixCAM 平台配置（工具链 URL、SHA-256、flags、prefix）：
  <https://github.com/sipeed/MaixCDK/blob/main/platforms/maixcam.yaml>
- MaixPy 的 MaixCAM 打包参数（Python 3.11、`linux_riscv64`、`musl_t-head`）：
  `D:\projects\MaixPy\setup.py`
- MaixPy RVV 文档（目标 3.11.6、构建 flags、`__riscv_vector` 检查）：
  `D:\projects\MaixPy\docs\doc\zh\pro\rvv.md`；
  在线：<https://github.com/sipeed/MaixPy/blob/main/docs/doc/zh/pro/rvv.md>
- `sg2002_rvv` README（3.11.6 / musl SDK 组件、构建注意事项）：
  <https://github.com/yangwenjie1231/sg2002_rvv>
- RVV 0.7.1 intrinsic 手册：
  `D:\projects\rvv-ext\Xuantie+900+Series+RVV-0.7.1+Intrinsic+Manual.pdf`
  - Preface p.1：使用 `v0p7`；
  - §20 p.19–24：0.7.1 与 1.0 的差异；
  - §18 p.19：`vdot` intrinsic 列表为 TODO。
- Sipeed MaixCAM 硬件规格：
  <https://wiki.sipeed.com/hardware/en/maixcam/maixcam.html>
- T-HEAD C906 规格（RVV 0.7.1 / VLEN 128）：
  <https://www.riscvschool.com/2023/03/09/t-head-xuantie-c906-risc-v/>
