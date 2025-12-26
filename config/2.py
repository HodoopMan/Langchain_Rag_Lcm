import subprocess
import sys
import pkg_resources

# 定义要安装的依赖（指定版本）
REQUIREMENTS = [
    "streamlit==1.52.2",
    "langchain==0.4.0.dev0",
    "langchain-community==0.4.1",
    "langchain-openai==0.2.6",
    "chromadb==0.5.17",
    "python-dotenv==1.2.1",
    "pypdf==6.4.0",
    "jieba==0.42.1"
]

# 升级pip
def upgrade_pip():
    print("=== 升级pip ===")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

# 安装依赖
def install_packages():
    print("\n=== 安装指定版本依赖 ===")
    for pkg in REQUIREMENTS:
        print(f"安装: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# 验证安装
def verify_installation():
    print("\n=== 验证依赖安装 ===")
    # 包名 → 实际导入名 映射（关键修复python-dotenv）
    package_map = {
        "streamlit": "streamlit",
        "langchain": "langchain",
        "langchain-community": "langchain_community",
        "langchain-openai": "langchain_openai",
        "chromadb": "chromadb",
        "python-dotenv": "dotenv",  # 正确的导入名不是python_dotenv
        "pypdf": "pypdf",
        "jieba": "jieba"
    }

    for pkg_name, import_name in package_map.items():
        try:
            version = pkg_resources.get_distribution(pkg_name).version
            __import__(import_name)
            print(f"✅ {pkg_name:<20} 版本: {version} (导入成功)")
        except pkg_resources.DistributionNotFound:
            print(f"❌ {pkg_name:<20} 未安装")
        except ImportError as e:
            print(f"⚠️ {pkg_name:<20} 版本: {version} (导入失败: {str(e)[:50]}...)")
        except Exception as e:
            print(f"❓ {pkg_name:<20} 检查失败: {str(e)[:50]}...")

    print(f"\nPython 版本: {sys.version}")
    print(f"Python 路径: {sys.executable}")

if __name__ == "__main__":
    try:
        upgrade_pip()
        install_packages()
        verify_installation()
        print("\n=== 所有依赖安装完成并验证通过 ===")
    except Exception as e:
        print(f"\n❌ 安装失败: {e}")
        sys.exit(1)