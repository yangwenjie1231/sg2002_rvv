# rvv (SG2002/C906 RVV accelerated NumPy ops)

This folder contains a small C++/pybind11 Python extension that exposes **NumPy-compatible** vector/matrix ops, with an **RVV 0.7.1** fast path for SG2002 (C906).

## Key goals

- Accept **NumPy `ndarray` / array-like** inputs directly.
- Internally convert inputs to **float32 + C-contiguous** buffers (the format RVV kernels expect).
- Use an RVV (RISC-V Vector Extension) implementation when available (`__riscv_vector`).
- Provide a scalar fallback so the same sources can be built/tested on non-RISC-V hosts.

## Build with Auto-bind-Python-from-CC--main (Linux)

This project uses the same build system as `Auto-bind-Python-from-CC--main/examples/demo`.

### 1) Open menuconfig (注意：命令是 menuconfig，不是 menuconfi)

```bash
python3 project.py menuconfig
```

### 2) Configure toolchain (SG2002 / C906FDV / musl t-head)

Important: **`CONFIG_TOOLCHAIN_PATH` must be an absolute path**.

If you set it to a relative path like:

- `host-tools/gcc/riscv64-linux-musl-x86_64/bin`

CMake will fail with:

- `TOOLCHAIN_PATH set error:host-tools/gcc/riscv64-linux-musl-x86_64/bin`

#### Recommended values

Using your toolchain metadata, set:

- `CONFIG_TOOLCHAIN_PATH` =
  `/root/xmusic/toolchains/maixcam/host-tools/gcc/riscv64-linux-musl-x86_64/bin`
  (adjust the prefix `/root/xmusic` to your real workspace path)
- `CONFIG_TOOLCHAIN_PREFIX` =
  `riscv64-unknown-linux-musl-`

Then select RISCV64 target (MaixCam).

This project’s `main/CMakeLists.txt` already injects the required RVV flags for SG2002 when `MaixCam` is enabled:

- `-mcpu=c906fdv`
- `-march=rv64imafdcv0p7xthead`
- `-mcmodel=medany`
- `-mabi=lp64d`

#### Alternative: configure without menuconfig

You can also configure via flags:

```bash
python3 project.py config \
  --toolchain /root/xmusic/toolchains/maixcam/host-tools/gcc/riscv64-linux-musl-x86_64/bin \
  --toolchain-prefix riscv64-unknown-linux-musl-
```

Or via environment variable used by this project’s top-level `CMakeLists.txt`:

```bash
export MY_TOOLCHAIN_PATH=/root/xmusic/toolchains/maixcam/host-tools/gcc/riscv64-linux-musl-x86_64/bin
```

### 3) Build + wheel

```bash
python3 project.py build
```

Artifacts:
- Compiled extension: `build/main/libmain.so`
- Wheel: `dist/*.whl`

### 4) Quick RVV enable check (optional)

After `python3 project.py build`, check the compile command line in the build log contains:

- `-march=rv64imafdcv0p7xthead`

If that flag is present, `__riscv_vector` should be defined and the RVV implementation (`rvv_rvv071.cpp`) will compile in.

## Python API

Top-level module name is `rvv`.

- `rvv.add(a, b) -> ndarray`
- `rvv.sub(a, b) -> ndarray`
- `rvv.scale(a, k) -> ndarray`
- `rvv.dot(a, b) -> float`
- `rvv.norm_l2(a) -> float`
- `rvv.matmul(A, B) -> ndarray`

All array inputs are converted internally to `float32` and C-contiguous.

## Tests

Tests are in `rvv_ext/tests` and require NumPy + pytest.

```bash
pytest -q
```

On SG2002, you can also run the micro-benchmark:

```bash
python3 tests/bench_perf.py
```
 
## Troubleshooting

### "Error: could not load cache" when running `cmake --build build ...`

This happens when the CMake cache (`build/CMakeCache.txt`) is missing.
Typical causes:
- You ran `python3 project.py distclean` and then did **not** re-run the CMake configure step.
- The `build/` directory was deleted or is empty.

Fix:

```bash
cd /path/to/rvv_ext

# Re-run configure (generates build/CMakeCache.txt)
python3 project.py build

# Then compile
cmake --build build --target main -- -j"$(nproc)"
```

### `setup.py` says `Native library not found: build/main/libmain.so`

It means you haven't built the extension yet (or it was built into a different folder).

Check:

```bash
find build -maxdepth 4 -name 'libmain.so' -o -name '*.so' | sort
```

If `libmain.so` exists but not under `build/main/`, adjust `SO_FILE_PATH` in `setup.py` accordingly.

## Copy-paste commands (embedded Linux)

### A) Build a wheel on the current machine (no NumPy required to *install*)

```bash
cd /home/ywj/ywj2/rvv_ext   # adjust to your real path

# (Optional) full clean: this removes build/ and CMakeCache.txt
python3 project.py distclean

# 1) Configure CMake (re-creates build/ and build/CMakeCache.txt)
python3 project.py build

# Sanity check (should exist after configure)
ls -la build/CMakeCache.txt

# 2) Compile the extension shared library (creates build/main/libmain.so)
cmake --build build --target main -- -j"$(nproc)"

# Sanity check (should exist after compile)
ls -la build/main/libmain.so

# 3) Build wheel (creates dist/*.whl)
python3 setup.py bdist_wheel

# 4) List wheel
ls -la dist
```

> If you only want to clean compiled objects but keep the CMake cache, prefer:
>
> ```bash
> python3 project.py clean
> ```
>
> instead of `distclean`.
