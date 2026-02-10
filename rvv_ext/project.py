#!/usr/bin/env python3
"""rvv_ext build entry.

This wrapper is needed because the upstream SDK build tool (`tools/cmake/project.py`) defaults
`sdk_path = ../../` when not injected, which can resolve to `/home/ywj` (or `/root`) on embedded
systems, causing errors like:

- FileNotFoundError: /home/ywj/tools/cmds

Behavior:
- Delegates to the SDK build script with an explicit `sdk_path`.
- Adds platform CMake args (-DLinux=ON / -DMaixCam=ON / -DMaixCam2=ON) based on generated config.
- Always passes -DSKIP_PROJECT_EXECUTABLE=ON (this project is a Python extension only).
- If CONFIG_BUILD_WHL_PACKAGE=y and command is build/rebuild, ensures `libmain.so` exists and
  runs `setup.py bdist_wheel` to produce `dist/*.whl`.

"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SDK_ENV_NAME = "MY_SDK_PATH"
CUSTOM_COMPONENTS_PATH_ENV_NAME = "CUSTOM_COMPONENTS_PATH"

CONFIG_BUILD_WHL_KEY = "CONFIG_BUILD_WHL_PACKAGE"

ARCH_CMAKE_MAP = {
    "CONFIG_TARGET_ARCH_X86": "Linux",
    "CONFIG_TARGET_ARCH_ARM64": "MaixCam2",
    "CONFIG_TARGET_ARCH_RISCV64": "MaixCam",
}


def _read_mk_config(path: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    if not path.exists():
        return cfg
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        cfg[k.strip()] = v.strip()
    return cfg


def _resolve_sdk_path(current_dir: Path) -> Path:
    env = os.environ.get(SDK_ENV_NAME)
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "tools" / "cmake" / "project.py").exists():
            return p

    # Common layouts:
    candidates = [
        current_dir / ".." / "Auto-bind-Python-from-CC--main" / "Auto-bind-Python-from-CC--main",
        current_dir / ".." / "Auto-bind-Python-from-CC--main",
        current_dir / ".." / ".." / "Auto-bind-Python-from-CC--main" / "Auto-bind-Python-from-CC--main",
    ]
    for c in candidates:
        c = c.resolve()
        if (c / "tools" / "cmake" / "project.py").exists():
            return c

    raise FileNotFoundError(
        "Cannot locate Auto-bind SDK. Set MY_SDK_PATH to the SDK root directory "
        "(the one containing tools/cmake/project.py)."
    )


def _resolve_custom_components_path() -> str | None:
    env = os.environ.get(CUSTOM_COMPONENTS_PATH_ENV_NAME)
    if not env:
        return None
    p = Path(env).expanduser().resolve()
    return str(p) if p.exists() else None


def _platform_cmake_args(config: dict[str, str]) -> list[str]:
    extra: list[str] = []
    for arch_key, cmake_var in ARCH_CMAKE_MAP.items():
        if config.get(arch_key) == "y":
            extra.append(f"-D{cmake_var}=ON")
            break
    # Always skip the framework executable (no main()).
    extra.append("-DSKIP_PROJECT_EXECUTABLE=ON")
    return extra


def _exec_file(path: Path, globals_dict: dict, work_dir: Path):
    old_cwd = os.getcwd()
    old_sys0 = sys.path[0]
    try:
        os.chdir(work_dir)
        sys.path[0] = str(work_dir)
        code = path.read_text(encoding="utf-8", errors="ignore")
        exec(compile(code, str(path), "exec"), globals_dict)
    finally:
        os.chdir(old_cwd)
        sys.path[0] = old_sys0


def _run_sdk_project(current_dir: Path, sdk_path: Path, custom_components_path: str | None, extra_cmake_args: list[str]):
    sdk_project = sdk_path / "tools" / "cmake" / "project.py"
    g = {
        "__builtins__": __builtins__,
        "__name__": "__main__",
        "__file__": str(sdk_project),
        "sdk_path": str(sdk_path),
        "custom_components_path": custom_components_path,
        "extra_cmake_args": extra_cmake_args,
    }
    _exec_file(sdk_project, g, current_dir)


def main() -> int:
    current_dir = Path(__file__).parent.resolve()

    # Safety: never run builds from inside the component folder.
    if current_dir.name == "main":
        raise SystemExit("Please run project.py from rvv_ext root, not from rvv_ext/main")

    # Read generated config (menuconfig output) if present.
    global_mk = current_dir / "build" / "config" / "global_config.mk"
    config = _read_mk_config(global_mk)

    sdk_path = _resolve_sdk_path(current_dir)

    # IMPORTANT: do NOT pass rvv_ext itself as CUSTOM_COMPONENTS_PATH.
    # If CUSTOM_COMPONENTS_PATH points to the project root, the SDK's component scanner
    # may pick up the `main/` component and then treat it as a standalone project,
    # causing recursive configure loops (PROJECT_PATH becomes .../rvv_ext/main).
    custom_components_path = None

    extra_cmake_args = _platform_cmake_args(config)

    cmd = sys.argv[1] if len(sys.argv) > 1 else None
    build_whl = config.get(CONFIG_BUILD_WHL_KEY) in ("y", "Y", "1", "true", "True")

    # Delegate to SDK project tool.
    _run_sdk_project(current_dir, sdk_path, custom_components_path, extra_cmake_args)

    # If wheel packaging enabled, ensure libmain.so exists and package.
    if cmd in ("build", "rebuild") and build_whl:
        libmain = current_dir / "build" / "main" / "libmain.so"
        if not libmain.exists():
            import subprocess

            jobs_n = os.cpu_count() or 2
            build_jobs = f"-j{jobs_n}"

            def _try(cmd_list: list[str]) -> bool:
                try:
                    subprocess.check_call(cmd_list, cwd=current_dir)
                    return True
                except Exception:
                    return False

            # Prefer building the default 'all' target.
            if not _try(["cmake", "--build", "build", "--", build_jobs]):
                # Fallback: try common explicit targets.
                _try(["cmake", "--build", "build", "--target", "main", "--", build_jobs])
                _try(["cmake", "--build", "build", "--target", "build_main_component", "--", build_jobs])

            # Final fallback: build inside component build dir
            if not libmain.exists():
                comp_dir = current_dir / "build" / "main"
                if (comp_dir / "Makefile").exists():
                    subprocess.check_call(["make", build_jobs], cwd=comp_dir)
                elif (comp_dir / "build.ninja").exists():
                    subprocess.check_call(["ninja", "-j", str(jobs_n)], cwd=comp_dir)

        if not libmain.exists():
            raise FileNotFoundError(f"Build succeeded but {libmain} not found")

        setup_py = current_dir / "setup.py"
        if not setup_py.exists():
            raise FileNotFoundError(f"setup.py not found: {setup_py}")

        # Run setup.py bdist_wheel
        old_argv = sys.argv
        try:
            sys.argv = ["setup.py", "bdist_wheel"]
            g = {"__builtins__": __builtins__, "__name__": "__main__", "__file__": str(setup_py)}
            _exec_file(setup_py, g, current_dir)
        finally:
            sys.argv = old_argv

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
