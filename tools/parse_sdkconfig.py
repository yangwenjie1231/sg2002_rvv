#!/usr/bin/env python3
import sys, re, os

def main(cfg):
    cxxflags = ["-std=c++17"]
    defs     = []
    with open(cfg) as f:
        for line in f:
            line = line.strip()
            if line.startswith("CONFIG_TOOLCHAIN_RISCV64=y"):
                cxxflags += ["-march=rv64gcv0p7", "-mrvv-vector-bits=128"]
            if line.startswith("CONFIG_EXTRA_CXXFLAGS="):
                m = re.search(r'"(.*)"', line)
                if m:
                    cxxflags.append(m.group(1))
    # 写出给 CMake 使用
    with open("compile_flags.txt","w") as f:
        f.write(" ".join(cxxflags))
    with open("compile_definitions.txt","w") as f:
        for d in defs:
            f.write(d + "\n")

if __name__ == "__main__":
    main(sys.argv[1])