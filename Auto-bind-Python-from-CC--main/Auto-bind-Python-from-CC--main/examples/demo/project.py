#!/usr/bin/env python3
"""
Build script for project preprocessing and compilation.
Handles configuration setup and delegates to project.py for actual build.
"""

import sys
import os
from pathlib import Path

# === Constants ===
SDK_ENV_NAME = "MY_SDK_PATH"
CUSTOM_COMPONENT_PATH_NAME = "CUSTOM_COMPONENTS_PATH"
CONFIG_KEY = "CONFIG_BUILD_WHL_PACKAGE"

# Architecture to CMake variable mapping
ARCH_CMAKE_MAP = {
    "CONFIG_TARGET_ARCH_X86": "Linux",
    "CONFIG_TARGET_ARCH_ARM64": "MaixCam2",
    "CONFIG_TARGET_ARCH_RISCV64": "MaixCam",
}


def get_current_dir():
    """Get directory where this script is located."""
    return Path(__file__).parent.resolve()


def resolve_sdk_path(current_dir):
    """Resolve SDK path from environment or default location."""
    env_path = os.environ.get(SDK_ENV_NAME)
    if env_path and Path(env_path).exists():
        return Path(env_path)
    return (current_dir / ".." / "..").resolve()


def resolve_custom_components_path():
    """Resolve custom components path from environment."""
    env_path = os.environ.get(CUSTOM_COMPONENT_PATH_NAME)
    if env_path and Path(env_path).exists():
        return Path(env_path)
    return None


def ensure_global_config(config_path):
    """Ensure global_config.mk exists, create if not."""
    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.touch()
        print(f"-- Created empty file: {config_path}")
    else:
        print(f"-- File already exists: {config_path}")


def load_config(config_path):
    """Load configuration from global_config.mk, return dict of enabled flags."""
    config = {}
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    config[key] = value.strip()
    except (IOError, OSError) as e:
        print(f"-- Warning: Failed to read config file: {e}")
    return config


def get_build_whl_flag(config):
    """Check if CONFIG_BUILD_WHL_PACKAGE is enabled."""
    return config.get(CONFIG_KEY, '') == 'y'


def get_platform_cmake_args(config):
    """Generate CMake args list based on architecture config."""
    for arch_key, cmake_var in ARCH_CMAKE_MAP.items():
        if config.get(arch_key, '') == 'y':
            print(f"-- Target architecture: {cmake_var}")
            return [f"-D{cmake_var}=ON"]
    print("-- Warning: No target architecture found in config!")
    return []


def exec_script(script_path, script_globals=None, work_dir=None):
    """Execute a Python script with proper namespace and working directory."""
    # 保存原始状态
    original_cwd = os.getcwd()
    original_sys_path_0 = sys.path[0]
    
    try:
        # 设置工作目录
        if work_dir:
            os.chdir(work_dir)
            sys.path[0] = str(work_dir)
        
        if script_globals is None:
            script_globals = {
                '__builtins__': __builtins__,
                '__name__': '__main__',
                '__file__': str(script_path),
            }
        
        with open(script_path, 'r', encoding='utf-8') as f:
            exec(f.read(), script_globals)
    finally:
        # 恢复原始状态
        os.chdir(original_cwd)
        sys.path[0] = original_sys_path_0


def build_whl_package(current_dir, sdk_path, extra_cmake_args):
    """Build wheel package workflow."""
    # 保存原始 sys.argv
    original_argv = sys.argv.copy()
    
    # Run cpp_bind_python.py (独立命名空间，在项目目录执行)
    cpp_bind_path = current_dir / "cpp_bind_python.py"
    if not cpp_bind_path.exists():
        print(f"-- Error: cpp_bind_python.py not found")
        return 1
    sys.argv = ['cpp_bind_python.py']
    exec_script(cpp_bind_path, work_dir=current_dir)
    
    # Run project.py from SDK (需要传入特定变量，在项目目录执行)
    sys.argv = original_argv
    project_path = sdk_path / "tools" / "cmake" / "project.py"
    if not project_path.exists():
        print(f"-- Error: project.py not found - {project_path}")
        return 1
    
    project_globals = {
        '__builtins__': __builtins__,
        '__name__': '__main__',
        '__file__': str(project_path),
        'sdk_path': str(sdk_path),
        'custom_components_path': None,
        'extra_cmake_args': extra_cmake_args,
    }
    exec_script(project_path, project_globals, work_dir=current_dir)
    
    # Run setup.py bdist_wheel (独立命名空间，在项目目录执行)
    setup_path = current_dir / "setup.py"
    if not setup_path.exists():
        print(f"-- Error: setup.py not found")
        return 1
    sys.argv = ['setup.py', 'bdist_wheel']
    exec_script(setup_path, work_dir=current_dir)
    
    # 恢复原始 sys.argv
    sys.argv = original_argv
    
    return 1  # 保持原代码的 sys.exit(1) 行为


def run_project(sdk_path, custom_components_path, extra_cmake_args, current_dir):
    """Run the main project.py script."""
    project_path = sdk_path / "tools" / "cmake" / "project.py"
    
    if not project_path.exists():
        print(f"-- Error: project.py not found - {project_path}")
        sys.exit(1)
    
    exec_globals = {
        '__builtins__': __builtins__,
        '__name__': '__main__',
        '__file__': str(project_path),
        'sdk_path': str(sdk_path),
        'custom_components_path': str(custom_components_path) if custom_components_path else None,
        'extra_cmake_args': extra_cmake_args,
    }
    
    try:
        exec_script(project_path, exec_globals, work_dir=current_dir)
    except SystemExit:
        raise
    except Exception as e:
        print(f"-- Error executing project.py: {e}")
        sys.exit(1)


def main():
    # Setup paths
    current_dir = get_current_dir()
    config_path = current_dir / "build" / "config" / "global_config.mk"
    sdk_path = resolve_sdk_path(current_dir)
    custom_components_path = resolve_custom_components_path()
    
    # Ensure config exists and load settings
    ensure_global_config(config_path)
    config = load_config(config_path)
    
    build_whl = get_build_whl_flag(config)
    extra_cmake_args = get_platform_cmake_args(config)
    
    # Print info
    print(f"CONFIG_BUILD_WHL_PACKAGE={1 if build_whl else 0}")
    print(f"-- SDK_PATH: {sdk_path}")
    print(f"-- CUSTOM_COMPONENTS_PATH: {custom_components_path}")
    if extra_cmake_args:
        print(f"-- Extra CMake args: {' '.join(extra_cmake_args)}")
    
    # Get command
    command = sys.argv[1] if len(sys.argv) >= 2 else None
    
    # Execute appropriate workflow
    if command in ('build', 'rebuild') and build_whl:
        sys.exit(build_whl_package(current_dir, sdk_path, extra_cmake_args))
    else:
        run_project(sdk_path, custom_components_path, extra_cmake_args, current_dir)


if __name__ == '__main__':
    main()