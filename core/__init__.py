# core/__init__.py
"""核心业务模块：包含文档加载、嵌入模型、向量库、问答链等核心逻辑"""
# 导出核心函数/类，简化外部导入（无需写 from core.embedding import xxx）
from .document_loader import load_documents, split_documents
from .embedding import init_siliconflow_embeddings
from .vector_store import create_vector_store
from .qa_chain import create_qa_chain, create_custom_prompt
from .similarity import get_similar_documents, is_irrelevant_question

# 声明对外暴露的接口（仅导出需要被外部调用的函数）
__all__ = [
    "load_documents",
    "split_documents",
    "init_siliconflow_embeddings",
    "create_vector_store",
    "create_qa_chain",
    "create_custom_prompt",
    "get_similar_documents",
    "is_irrelevant_question"
]