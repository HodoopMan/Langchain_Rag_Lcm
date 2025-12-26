"""日志配置工具"""
import logging
from config.settings import Config

def setup_logger(name: str = "tcm_rag") -> logging.Logger:
    """配置日志"""
    logging.basicConfig(
        level=Config.LOG_CONFIG["level"],
        format=Config.LOG_CONFIG["format"],
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(name)
    return logger

# 全局日志实例
logger = setup_logger()