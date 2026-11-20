# 与「编译.md」一致的交叉/本机工具链模板
set(CMAKE_SYSTEM_NAME Linux)

# 根据 sdkconfig 自动切换
if(DEFINED CONFIG_TOOLCHAIN_THEAD_MUSL)
    set(CMAKE_SYSTEM_PROCESSOR riscv64)
    set(CMAKE_C_COMPILER   riscv64-unknown-linux-musl-gcc)
    set(CMAKE_CXX_COMPILER riscv64-unknown-linux-musl-g++)
    set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
    add_definitions(-DCONFIG_TOOLCHAIN_THEAD_MUSL)
    # 添加 T-Head 特定的编译选项
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=c906fdv -march=rv64imafdcv0p7xthead -mcmodel=medany -mabi=lp64d")
elseif(DEFINED CONFIG_TOOLCHAIN_RISCV64)
    set(CMAKE_SYSTEM_PROCESSOR riscv64)
    set(CMAKE_C_COMPILER   riscv64-unknown-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER riscv64-unknown-linux-gnu-g++)
    add_definitions(-DCONFIG_TOOLCHAIN_RISCV64)
elseif(DEFINED CONFIG_TOOLCHAIN_NATIVE)
    set(CMAKE_SYSTEM_PROCESSOR x86_64)
    # 使用系统默认 gcc/g++
endif()