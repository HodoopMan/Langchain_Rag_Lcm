"""相似度检索模块"""
from typing import List, Tuple
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from utils.logger import logger


def get_similar_documents(
        vector_store: Chroma,
        query: str,
        k: int = 5
) -> List[Tuple[Document, float]]:
    """获取带相似度评分的检索结果"""
    if not vector_store:
        logger.warning("向量库未初始化，无法进行相似度检索")
        return []

    try:
        similar_docs = vector_store.similarity_search_with_score(
            query=query,
            k=k
        )
        # 按相似度降序排序
        similar_docs.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"相似度检索完成，找到{len(similar_docs)}个相关文档")
        return similar_docs
    except Exception as e:
        logger.error(f"相似度检索失败：{str(e)}", exc_info=True)
        return []


def is_irrelevant_question(
        vector_store: Chroma,
        query: str,
        threshold: float = 0.5
) -> bool:
    """判断是否为无关问题（基于相似度阈值）"""
    similar_docs = get_similar_documents(vector_store, query)
    if not similar_docs:
        return True

    max_score = max([score for _, score in similar_docs])
    return max_score < threshold