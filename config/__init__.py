# config/__init__.py
"""配置模块：提供项目核心配置类"""
# 导出核心配置类，让外部可以直接 from config import Config
from .settings import Config

# 声明对外暴露的接口（规范写法）
__all__ = ["Config"]