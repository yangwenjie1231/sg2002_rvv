# Default config for wheel packaging
CONFIG_BUILD_WHL_PACKAGE=y

# Kconfig string values must be quoted, otherwise genconfig.py will warn
# "malformed string literal" and ignore the assignment.
CONFIG_WHL_PACKAGE_NAME="rvv"
CONFIG_WHL_VERSION="0.1.0"
CONFIG_WHL_AUTHOR="xmusic"
CONFIG_WHL_DESCRIPTION="RVV accelerated NumPy ops (NumPy-compatible)"

# Target selection (choose ONE in menuconfig).
# Keep empty here so menuconfig choice wins.
# CONFIG_TARGET_ARCH_X86=y
# CONFIG_TARGET_ARCH_ARM64=y
# CONFIG_TARGET_ARCH_RISCV64=y

# Toolchain configuration (recommended to set via menuconfig or:
#   python3 project.py config --toolchain <ABS_PATH> --toolchain-prefix <PREFIX>
# )
# CONFIG_TOOLCHAIN_PATH="/abs/path/to/.../riscv64-linux-musl-x86_64/bin"
# CONFIG_TOOLCHAIN_PREFIX="riscv64-unknown-linux-musl-"
