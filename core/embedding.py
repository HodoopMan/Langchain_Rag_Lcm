"""嵌入模型初始化模块"""
from langchain_openai import OpenAIEmbeddings
from config.settings import Config
from utils.logger import logger

def init_siliconflow_embeddings(model_name: str) -> OpenAIEmbeddings:
    """初始化硅基流动嵌入模型"""
    api_key = Config.SILICONFLOW_CONFIG["api_key"]
    if not api_key:
        logger.error("硅基流动API Key未配置")
        return None

    try:
        embeddings = OpenAIEmbeddings(
            model=model_name,
            api_key=api_key,
            base_url=Config.SILICONFLOW_CONFIG["base_url"],
            chunk_size=Config.SILICONFLOW_CONFIG["chunk_size"]
        )

        # 验证模型
        test_embedding = embeddings.embed_query("感冒的中医治疗")
        logger.info(f"硅基流动嵌入模型初始化成功：{model_name}，向量维度：{len(test_embedding)}")
        return embeddings

    except Exception as e:
        logger.error(f"初始化硅基流动嵌入模型失败：{str(e)}", exc_info=True)
        return None