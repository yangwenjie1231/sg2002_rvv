#!/usr/bin/env python3
import sys, re, os

def main(cfg):
    cxxflags = ["-std=c++17"]
    defs     = []
    with open(cfg, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            # 工具链
            if 'CONFIG_TOOLCHAIN_THEAD_MUSL' in line and line.endswith('=y'):
                cxxflags += ['-march=rv64gcv0p7', '-mrvv-vector-bits=128', '-mcpu=c906fdv', '-march=rv64imafdcv0p7xthead', '-mcmodel=medany', '-mabi=lp64d']
            if 'CONFIG_TOOLCHAIN_RISCV64' in line and line.endswith('=y'):
                cxxflags += ['-march=rv64gcv0p7', '-mrvv-vector-bits=128']
            if 'CONFIG_TOOLCHAIN_NATIVE' in line and line.endswith('=y'):
                pass   # 本机编译，无需额外 flags
            # 额外 flags
            m = re.match(r'CONFIG_EXTRA_CXXFLAGS="(.+)"', line)
            if m:
                cxxflags.append(m.group(1))
    # 写出给 CMake 使用
    with open('compile_flags.txt', 'w', encoding='utf-8') as f:
        f.write(' '.join(cxxflags))
    with open('compile_definitions.txt', 'w', encoding='utf-8') as f:
        for d in defs:
            f.write(d + '\n')

if __name__ == "__main__":
    main(sys.argv[1])