import os,sys
import shutil
from setuptools import setup, find_packages
import re 

# ==============================================
# 解析 global_config.mk 配置文件
# ==============================================
def parse_mk_config(file_path):
    """解析 .mk 配置文件，返回键值对字典"""
    config = {}
    if not os.path.exists(file_path):
        print(f"❌ 错误：找不到配置文件 {file_path}")
        sys.exit(1)
    
    # 正则表达式匹配 .mk 文件中的赋值语句（支持带引号的值）
    pattern = re.compile(r'^(\w+)\s*=\s*(.+?)\s*$')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            # 跳过空行和注释行
            if not line or line.startswith('#'):
                continue
            
            match = pattern.match(line)
            if match:
                key = match.group(1)
                value = match.group(2)
                # 去除值两端的引号（支持单/双引号）
                if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                
                # 特殊处理：KEYWORDS 和 PROJECT_URLS 转为列表/字典
                if key == "CONFIG_WHL_KEYWORDS":
                    # 格式示例："add,math,calculation" -> ["add", "math", "calculation"]
                    value = [k.strip() for k in value.split(',')] if value else []
                elif key == "CONFIG_WHL_PROJECT_URLS":
                    # 格式示例："Homepage:https://xxx,Documentation:https://yyy" -> {"Homepage": "https://xxx", ...}
                    project_urls = {}
                    if value:
                        pairs = [p.strip() for p in value.split(',')]
                        for pair in pairs:
                            if ':' in pair:
                                k, v = pair.split(':', 1)
                                project_urls[k.strip()] = v.strip()
                    value = project_urls
                
                config[key] = value
    
    return config

# 读取配置文件
CONFIG_FILE = os.path.join("build", "config", "global_config.mk")
mk_config = parse_mk_config(CONFIG_FILE)

# 1. 配置基本信息（自动读取）
PACKAGE_NAME = mk_config.get("CONFIG_WHL_PACKAGE_NAME", "add") 
VERSION = mk_config.get("CONFIG_WHL_VERSION", "0.1.0") 
AUTHOR = mk_config.get("CONFIG_WHL_AUTHOR", "Tao")
DESCRIPTION = mk_config.get("CONFIG_WHL_DESCRIPTION", "Auto bind Python from C/C++")

# 2. 路径配置（目标目录为 build/模块名）
SO_FILE_PATH = os.path.join("build", "main", "libmain.so")  # 原始.so文件路径
# 新的包目录：根目录/build/模块名
PACKAGE_DIR = os.path.join("build", PACKAGE_NAME)          
TARGET_SO_NAME = f"{PACKAGE_NAME}.so"                       # 重命名后的.so文件名
TARGET_SO_PATH = os.path.join(PACKAGE_DIR, TARGET_SO_NAME)  # .so文件最终位置
INIT_FILE_PATH = os.path.join(PACKAGE_DIR, "__init__.py")   # __init__.py路径

# 3. 核心自动化函数
def prepare_package():
    """自动准备包目录、__init__.py、复制.so文件"""
    # 第一步：创建包目录（如果不存在，包含build父目录）
    if not os.path.exists(PACKAGE_DIR):
        os.makedirs(PACKAGE_DIR)
        print(f"✅ 已创建包目录：{PACKAGE_DIR}")

    # 第二步：自动生成__init__.py文件
    init_content = f'''"""
{PACKAGE_NAME} 包 - {DESCRIPTION}
作者：{AUTHOR}
版本：{VERSION}
"""
# 自动生成的__init__.py，用于导入.so模块的核心功能
from .{PACKAGE_NAME} import *

# 暴露版本信息，方便用户查看
__version__ = "{VERSION}"
__author__ = "{AUTHOR}"
'''
    # 写入__init__.py（覆盖已有文件，确保内容最新）
    with open(INIT_FILE_PATH, "w", encoding="utf-8") as f:
        f.write(init_content)
    print(f"✅ 已自动生成 {INIT_FILE_PATH}")

    # 第三步：复制并重命名.so文件到包目录
    if not os.path.exists(SO_FILE_PATH):
        raise FileNotFoundError(
            f"❌ 错误：找不到.so文件！路径：{SO_FILE_PATH}\n"
            "请先编译生成libmain.so，再执行打包命令。"
        )
    shutil.copy(SO_FILE_PATH, TARGET_SO_PATH)
    print(f"✅ 已复制.so文件到：{TARGET_SO_PATH}")

# 4. 执行前置准备（创建目录、生成__init__.py、复制.so）
prepare_package()

# 5. 打包配置（适配新的包目录路径）
setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    description=DESCRIPTION,
    # 指定包的根目录为build，自动查找build下的模块包
    package_dir={"": "build"},
    # 仅打包声明的目录包（从build目录下找PACKAGE_NAME）
    packages=find_packages(where="build"),
    # 打包包内的.so文件
    package_data={
        PACKAGE_NAME: [TARGET_SO_NAME]  # 相对于包目录（build/add）的路径
    },
    # 兼容配置
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Linux",
        "Programming Language :: C++",
    ],
    python_requires=">=3.6",
)


print(f"🎉 打包完成,在dist目录下,包名:{PACKAGE_NAME},版本{VERSION}")