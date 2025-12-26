#%%
import pkg_resources

# 定义需要检查的包列表（包含可选包）
packages_to_check = [
    "streamlit",
    "langchain",
    "langchain-community",
    "langchain-openai",
    "chromadb",
    "python-dotenv",
    "pypdf",
    "jieba"
]

print("=" * 50)
print("Python 依赖包版本信息")
print("=" * 50)

for package in packages_to_check:
    try:
        # 处理带连字符的包名（转换为导入名）
        import_name = package.replace("-", "_") if "-" in package else package

        # 获取版本号
        version = pkg_resources.get_distribution(package).version

        # 尝试导入验证
        __import__(import_name)

        print(f"✅ {package:<20} 版本: {version}")
    except pkg_resources.DistributionNotFound:
        print(f"❌ {package:<20} 未安装")
    except ImportError as e:
        print(f"⚠️ {package:<20} 版本: {version} (导入失败: {str(e)[:50]}...)")
    except Exception as e:
        print(f"❓ {package:<20} 检查失败: {str(e)[:50]}...")

# 额外打印Python版本
import sys
print("\n" + "=" * 50)
print(f"Python 版本: {sys.version}")
print(f"Python 解释器路径: {sys.executable}")
print("=" * 50)