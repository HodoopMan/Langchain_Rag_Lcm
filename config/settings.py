"""核心配置类"""
from pathlib import Path
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()


class Config:
    """中医RAG配置类"""
    # 基础路径配置
    BASE_DIR = Path(__file__).parent.parent
    DOC_PATH = Path(os.getenv("DOC_PATH", BASE_DIR / "./docs/疾病.txt"))
    VECTOR_STORE_DIR = Path(os.getenv("VECTOR_STORE_DIR", BASE_DIR / "chroma_db/traditional_chinese_medicine"))

    # 文本分割配置
    TEXT_SPLITTER_CONFIG = {
        "chunk_size": 400,
        "chunk_overlap": 40,
        "separators": ["\n\n", "\n", "。", "！", "？", "；", "：", "，", "、", "（", "）", "【", "】", "——", " ", ""]
    }

    # 硅基流动配置
    SILICONFLOW_CONFIG = {
        "api_key": os.getenv("SILICONFLOW_API_KEY", ""),
        "base_url": os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        "embedding_model": "BAAI/bge-m3",
        "llm_model": "Qwen/Qwen2-7B-Instruct",
        "temperature": 0,
        "max_tokens": 2000,
        "chunk_size": 1000
    }

    # 检索配置
    RETRIEVER_CONFIG = {
        "k": 5,
        "search_type": "similarity"
    }

    # 嵌入模型列表
    EMBEDDING_MODELS = [
        "BAAI/bge-m3",
        "BAAI/bge-large-zh-v1.5",
        "text-embedding-ada-002",
        "moka-ai/m3e-large",
        "intfloat/multilingual-e5-large"
    ]

    # LLM模型列表
    LLM_MODELS = [
        "Qwen/Qwen2-7B-Instruct",
        "Qwen/Qwen2-14B-Instruct",
        "Meta-Llama-3-8B-Instruct"
    ]

    # 提示词模板
    PROMPT_TEMPLATE = """
    你是一位资深的中医临床诊疗专家，精通中医理论和临床实践，尤其擅长各类中医疾病的辨证论治。
    请根据以下规则回答用户的问题：
    1. 必须基于提供的文档内容回答，确保专业性和准确性；

    参考文档内容：
    {context}

    用户当前问题：
    {question}
    """

    # 无关问题回复
    IRRELEVANT_REPLY = "抱歉，我专注于中医临床诊疗领域。如需了解中医疾病的诊疗方法，我会尽力解答。"

    # 上下文记忆配置
    CONTEXT_MEMORY_CONFIG = {
        "max_history_length": 10,
        "history_separator": "\n---\n",
        "context_window_size": 3000
    }

    # 日志配置
    LOG_CONFIG = {
        "level": os.getenv("LOG_LEVEL", "INFO"),
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }