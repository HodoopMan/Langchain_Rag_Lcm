"""文档加载与分割模块"""
from typing import List
from pathlib import Path
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.settings import Config
from utils.logger import logger

def load_documents(file_path: Path) -> List[Document]:
    """加载文档（支持多编码格式）"""
    documents = []
    if not file_path.exists():
        logger.error(f"文档文件不存在：{file_path.absolute()}")
        return documents

    encodings = ["utf-8", "gbk", "gb2312", "utf-8-sig"]
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
                doc_text = "".join(lines).strip()

            if not doc_text:
                logger.warning("文档内容为空")
                return documents

            document = Document(
                page_content=doc_text,
                metadata={
                    "source": str(file_path.absolute()),
                    "file_name": file_path.name,
                    "file_size": file_path.stat().st_size,
                    "encoding": encoding
                }
            )
            documents.append(document)
            logger.info(f"成功加载文档：{file_path.name}（编码：{encoding}）")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"加载文档失败：{str(e)}")
            return documents
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """分割文档为文本块"""
    if not documents:
        return []
    text_splitter = RecursiveCharacterTextSplitter(**Config.TEXT_SPLITTER_CONFIG)
    split_docs = text_splitter.split_documents(documents)
    logger.info(f"文本切割完成：原始{len(documents)}个文档 → 切割后{len(split_docs)}个文本块")
    return split_docs