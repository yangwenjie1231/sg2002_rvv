# 极简交叉编译工具链示例（RISC‑64）
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSORriscv64)

set(CMAKE_C_COMPILER riscv64-linux-gnu-g++)
set(CMAKE_CXX_COMPILER riscv64-linux-gnu-g++)

# 如果 sdkconfig 已给出 march，这里不再重复
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64gcv0p7")