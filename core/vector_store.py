"""向量库管理模块"""
from typing import List
from pathlib import Path
import shutil
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from config.settings import Config
from utils.logger import logger

def create_vector_store(
    split_docs: List[Document],
    embeddings,
    reset: bool = False
) -> Chroma:
    """创建/重置Chroma向量库"""
    if not split_docs:
        logger.error("切割后的文本块为空，无法创建向量库")
        return None

    if not embeddings:
        logger.error("嵌入模型未初始化，无法创建向量库")
        return None

    # 确保存储目录存在
    vector_store_dir = Config.VECTOR_STORE_DIR
    vector_store_dir.mkdir(parents=True, exist_ok=True)

    # 重置向量库
    if reset or vector_store_dir.exists():
        try:
            shutil.rmtree(vector_store_dir)
            vector_store_dir.mkdir(parents=True, exist_ok=True)
            logger.warning("已重置向量库（保证嵌入模型维度一致）")
        except Exception as e:
            logger.error(f"重置向量库失败：{str(e)}")
            return None

    try:
        vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=str(vector_store_dir)
        )
        vector_store.persist()
        logger.info(f"向量库创建完成（存储路径：{vector_store_dir}）")
        return vector_store
    except Exception as e:
        logger.error(f"创建向量库失败：{str(e)}", exc_info=True)
        return None