"""问答链构建模块"""
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from config.settings import Config
from utils.logger import logger

def create_custom_prompt() -> PromptTemplate:
    """创建自定义提示词模板"""
    prompt = PromptTemplate(
        template=Config.PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    return prompt

def create_qa_chain(vector_store: Chroma) -> ConversationalRetrievalChain:
    """创建带上下文记忆的问答链"""
    if not vector_store:
        logger.error("向量库未初始化，无法创建问答链")
        return None

    api_key = Config.SILICONFLOW_CONFIG["api_key"]
    if not api_key:
        logger.error("硅基流动API Key未配置")
        return None

    try:
        # 初始化LLM
        llm = ChatOpenAI(
            model=Config.SILICONFLOW_CONFIG["llm_model"],
            api_key=api_key,
            base_url=Config.SILICONFLOW_CONFIG["base_url"],
            temperature=Config.SILICONFLOW_CONFIG["temperature"],
            max_tokens=Config.SILICONFLOW_CONFIG["max_tokens"]
        )
        logger.info(f"LLM初始化成功：{Config.SILICONFLOW_CONFIG['llm_model']}")

        # 创建检索器
        retriever = vector_store.as_retriever(
            search_type=Config.RETRIEVER_CONFIG["search_type"],
            search_kwargs={"k": Config.RETRIEVER_CONFIG["k"]}
        )

        # 创建自定义提示词
        custom_prompt = create_custom_prompt()

        # 创建问答链
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
            verbose=False,
            combine_docs_chain_kwargs={
                "prompt": custom_prompt,
                "document_variable_name": "context"
            },
            max_tokens_limit=Config.CONTEXT_MEMORY_CONFIG["context_window_size"],
            output_key="answer"
        )

        logger.info("问答链初始化完成（带上下文记忆）")
        return qa_chain
    except Exception as e:
        logger.error(f"创建问答链失败：{str(e)}", exc_info=True)
        return None