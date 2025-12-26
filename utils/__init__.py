# utils/__init__.py
"""工具模块：提供日志配置、对话历史处理等通用工具函数"""
# 导出工具函数/实例
from .logger import setup_logger, logger
from .chat_history import format_chat_history_for_display, trim_chat_history

# 声明对外暴露的接口
__all__ = [
    "setup_logger",
    "logger",
    "format_chat_history_for_display",
    "trim_chat_history"
]