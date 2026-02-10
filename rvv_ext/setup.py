import os
import re
import shutil
import sys

from setuptools import find_packages, setup


def parse_mk_config(file_path: str) -> dict:
    """Parse build/config/global_config.mk into a dict."""
    config: dict = {}
    if not os.path.exists(file_path):
        print(f"config file not found: {file_path}")
        sys.exit(1)

    pattern = re.compile(r'^(\w+)\s*=\s*(.+?)\s*$')
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = pattern.match(line)
            if not m:
                continue
            key, value = m.group(1), m.group(2)
            if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            config[key] = value

    return config


CONFIG_FILE = os.path.join("build", "config", "global_config.mk")
mk = parse_mk_config(CONFIG_FILE)

PACKAGE_NAME = mk.get("CONFIG_WHL_PACKAGE_NAME", "rvv")
VERSION = mk.get("CONFIG_WHL_VERSION", "0.1.0")
AUTHOR = mk.get("CONFIG_WHL_AUTHOR", "xmusic")
DESCRIPTION = mk.get("CONFIG_WHL_DESCRIPTION", "RVV accelerated NumPy ops")

SO_FILE_PATH = os.path.join("build", "main", "libmain.so")
PACKAGE_DIR = os.path.join("build", PACKAGE_NAME)
TARGET_SO_NAME = f"{PACKAGE_NAME}.so"
TARGET_SO_PATH = os.path.join(PACKAGE_DIR, TARGET_SO_NAME)
INIT_FILE_PATH = os.path.join(PACKAGE_DIR, "__init__.py")

# Type hint files (optional but recommended for IDE completion)
STUB_FILE_SRC = os.path.join(os.path.dirname(__file__), f"{PACKAGE_NAME}.pyi")
PY_TYPED_SRC = os.path.join(os.path.dirname(__file__), "py.typed")
STUB_FILE_DST = os.path.join(PACKAGE_DIR, f"{PACKAGE_NAME}.pyi")
PY_TYPED_DST = os.path.join(PACKAGE_DIR, "py.typed")


def prepare_package():
    os.makedirs(PACKAGE_DIR, exist_ok=True)

    init_content = f'''"""{PACKAGE_NAME} - {DESCRIPTION}

This package is generated during the wheel build.

The API is implemented in the native extension `{PACKAGE_NAME}.so`.
"""

from .{PACKAGE_NAME} import *  # noqa: F401,F403

__version__ = "{VERSION}"
__author__ = "{AUTHOR}"
'''
    with open(INIT_FILE_PATH, "w", encoding="utf-8") as f:
        f.write(init_content)

    if not os.path.exists(SO_FILE_PATH):
        raise FileNotFoundError(
            f"Native library not found: {SO_FILE_PATH}. "
            "Build first so CMake produces build/main/libmain.so"
        )

    shutil.copy(SO_FILE_PATH, TARGET_SO_PATH)

    # Copy type hints if present.
    if os.path.exists(STUB_FILE_SRC):
        shutil.copy(STUB_FILE_SRC, STUB_FILE_DST)
    if os.path.exists(PY_TYPED_SRC):
        shutil.copy(PY_TYPED_SRC, PY_TYPED_DST)


prepare_package()

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    description=DESCRIPTION,
    package_dir={"": "build"},
    packages=find_packages(where="build"),
    package_data={PACKAGE_NAME: [TARGET_SO_NAME, f"{PACKAGE_NAME}.pyi", "py.typed"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Linux",
        "Programming Language :: C++",
    ],
    python_requires=">=3.8",
    install_requires=[],
)
