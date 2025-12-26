import os
import sys
print("Python解释器路径：", sys.executable)
print("模块搜索路径：", sys.path[:3])  # 只看前3个关键路径
import logging
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

# LangChain相关导入
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()


# ===================== 核心配置 =====================
class Config:
    """中医RAG配置类（硅基流动嵌入模型版）"""
    DOC_PATH = Path("./docs/疾病.txt")
    VECTOR_STORE_DIR = Path("./chroma_db/traditional_chinese_medicine")

    # 文本分割配置
    TEXT_SPLITTER_CONFIG = {
        "chunk_size": 400,
        "chunk_overlap": 40,
        "separators": ["\n\n", "\n", "。", "！", "？", "；", "：", "，", "、", "（", "）", "【", "】", "——", " ", ""]
    }

    # 硅基流动嵌入模型配置
    SILICONFLOW_EMBEDDING_CONFIG = {
        "model": "BAAI/bge-m3",
        "api_key": "sk-tvamliktbquwrcikvlxmwmdkwlumgfnstidlzudbzwmajwgf",
        "base_url": "https://api.siliconflow.cn/v1",
        "chunk_size": 1000,
    }

    # 硅基流动LLM配置
    SILICONFLOW_LLM_CONFIG = {
        "model": "Qwen/Qwen2-7B-Instruct",
        "api_key": "sk-tvamliktbquwrcikvlxmwmdkwlumgfnstidlzudbzwmajwgf",
        "base_url": "https://api.siliconflow.cn/v1",
        "temperature": 0,
        "max_tokens": 2000
    }

    # 检索配置
    RETRIEVER_CONFIG = {
        "k": 5,
        "search_type": "similarity"
    }

    # 硅基流动支持的嵌入模型列表
    EMBEDDING_MODELS = [
        "BAAI/bge-m3",
        "BAAI/bge-large-zh-v1.5",
        "text-embedding-ada-002",
        "moka-ai/m3e-large",
        "intfloat/multilingual-e5-large"
    ]

    # 🔥 修复：简化提示词模板（移除chat_history变量，由Chain自动处理）
    PROMPT_TEMPLATE = """
    你是一位资深的中医临床诊疗专家，精通中医理论和临床实践，尤其擅长各类中医疾病的辨证论治。
    请根据以下规则回答用户的问题：
    1. 必须基于提供的文档内容回答，确保专业性和准确性；

    参考文档内容：
    {context}

    用户当前问题：
    {question}
    """

    # 系统提示词
    SYSTEM_PROMPT = """
    你是中医临床诊疗RAG助手，具备以下能力：
    1. 精准理解中医术语和临床问题
    2. 结合历史对话上下文，保持回答的连贯性
    3. 记忆用户之前的提问和关注点
    4. 用户输入了与中医无关的问直接回答:抱歉，我专注于中医临床诊疗领域。如需了解中医疾病的诊疗方法，我会尽力解答。
    """

    # 上下文记忆配置
    CONTEXT_MEMORY_CONFIG = {
        "max_history_length": 10,
        "history_separator": "\n---\n",
        "context_window_size": 3000
    }


# ===================== 工具函数（修复对话历史格式） =====================
def format_chat_history_for_display(chat_history: List[Tuple[str, str]]) -> str:
    """格式化对话历史用于展示（仅前端显示用）"""
    if not chat_history:
        return "无"

    # 限制历史长度
    history_to_use = chat_history[-Config.CONTEXT_MEMORY_CONFIG["max_history_length"]:]

    formatted_history = []
    for i, (human, ai) in enumerate(history_to_use, 1):
        formatted_history.append(f"用户{i}：{human}")
        formatted_history.append(f"助手{i}：{ai}")

    return Config.CONTEXT_MEMORY_CONFIG["history_separator"].join(formatted_history)


def create_custom_prompt() -> PromptTemplate:
    """创建自定义提示词模板（修复：仅保留context和question变量）"""
    prompt = PromptTemplate(
        template=Config.PROMPT_TEMPLATE,
        input_variables=["context", "question"]  # 移除chat_history变量
    )
    return prompt


@st.cache_resource(show_spinner="正在加载中医文档...")
def load_documents(file_path: Path) -> List[Document]:
    documents = []
    if not file_path.exists():
        st.error(f"文档文件不存在：{file_path.absolute()}")
        return documents

    encodings = ["utf-8", "gbk", "gb2312", "utf-8-sig"]
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
                doc_text = "".join(lines).strip()

            if not doc_text:
                st.warning("文档内容为空")
                return documents

            document = Document(
                page_content=doc_text,
                metadata={"source": str(file_path.absolute()), "file_name": file_path.name,
                          "file_size": file_path.stat().st_size, "encoding": encoding}
            )
            documents.append(document)
            st.success(f"成功加载文档：{file_path.name}（编码：{encoding}）")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            st.error(f"加载文档失败：{str(e)}")
            return documents
    return documents


@st.cache_resource(show_spinner="正在切割文本...")
def split_documents(documents: List[Document]) -> List[Document]:
    if not documents:
        return []
    text_splitter = RecursiveCharacterTextSplitter(**Config.TEXT_SPLITTER_CONFIG)
    split_docs = text_splitter.split_documents(documents)
    st.info(f"文本切割完成：原始{len(documents)}个文档 → 切割后{len(split_docs)}个文本块")
    return split_docs


@st.cache_resource(show_spinner="正在初始化硅基流动嵌入模型...")
def init_siliconflow_embeddings(model_name: str) -> OpenAIEmbeddings:
    """初始化硅基流动嵌入模型"""
    if not Config.SILICONFLOW_EMBEDDING_CONFIG["api_key"]:
        st.error("硅基流动API Key未配置，请在.env文件中设置SILICONFLOW_API_KEY")
        return None

    try:
        embeddings = OpenAIEmbeddings(
            model=model_name,
            api_key=Config.SILICONFLOW_EMBEDDING_CONFIG["api_key"],
            base_url=Config.SILICONFLOW_EMBEDDING_CONFIG["base_url"],
            chunk_size=Config.SILICONFLOW_EMBEDDING_CONFIG["chunk_size"]
        )

        # 验证模型
        test_embedding = embeddings.embed_query("感冒的中医治疗")
        st.success(f"✅ 硅基流动嵌入模型初始化成功：{model_name}")
        st.info(f"📏 向量维度：{len(test_embedding)}")
        return embeddings

    except Exception as e:
        st.error(f"初始化硅基流动嵌入模型失败：{str(e)}")
        logger.error(f"嵌入模型初始化错误：{str(e)}", exc_info=True)
        return None


@st.cache_resource(show_spinner="正在创建向量库...")
def create_vector_store(split_docs: List[Document], embeddings, reset: bool = False) -> Chroma:
    if not split_docs:
        st.error("切割后的文本块为空，无法创建向量库")
        return None

    if not embeddings:
        st.error("嵌入模型未初始化，无法创建向量库")
        return None

    # 强制重置向量库
    Config.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    if reset or Config.VECTOR_STORE_DIR.exists():
        import shutil
        try:
            shutil.rmtree(Config.VECTOR_STORE_DIR)
            Config.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
            st.warning("🗑️ 已重置向量库（保证嵌入模型维度一致）")
        except Exception as e:
            st.error(f"重置向量库失败：{str(e)}")
            return None

    try:
        vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=str(Config.VECTOR_STORE_DIR)
        )
        vector_store.persist()
        st.success(f"✅ 向量库创建完成（存储路径：{Config.VECTOR_STORE_DIR}）")
        return vector_store
    except Exception as e:
        st.error(f"创建向量库失败：{str(e)}")
        return None


# 相似度检索
def get_similar_documents(vector_store: Chroma, query: str, k: int = 5) -> List[Tuple[Document, float]]:
    """获取带相似度的检索结果"""
    if not vector_store:
        return []

    similar_docs = vector_store.similarity_search_with_score(
        query=query,
        k=k
    )
    similar_docs.sort(key=lambda x: x[1], reverse=True)
    return similar_docs


# 🔥 核心修复：调整问答链创建方式
@st.cache_resource(show_spinner="正在初始化硅基流动LLM...")
def create_qa_chain(_vector_store: Chroma) -> ConversationalRetrievalChain:
    """创建带上下文记忆的问答链（修复对话历史格式问题）"""
    if not _vector_store:
        return None

    if not Config.SILICONFLOW_LLM_CONFIG["api_key"]:
        st.error("硅基流动API Key未配置，请检查.env文件")
        return None

    try:
        # 初始化硅基流动LLM
        llm = ChatOpenAI(**Config.SILICONFLOW_LLM_CONFIG)
        st.success(f"✅ 硅基流动LLM初始化成功：{Config.SILICONFLOW_LLM_CONFIG['model']}")
    except Exception as e:
        st.error(f"初始化LLM失败：{str(e)}")
        return None

    # 创建检索器
    retriever = _vector_store.as_retriever(
        search_type=Config.RETRIEVER_CONFIG["search_type"],
        search_kwargs={"k": Config.RETRIEVER_CONFIG["k"]}
    )

    # 创建自定义提示词
    custom_prompt = create_custom_prompt()

    # 🔥 修复：创建问答链（使用Chain默认的chat_history处理方式）
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        verbose=False,
        # 使用自定义提示词
        combine_docs_chain_kwargs={
            "prompt": custom_prompt,
            "document_variable_name": "context"
        },
        # 上下文配置
        max_tokens_limit=Config.CONTEXT_MEMORY_CONFIG["context_window_size"],
        # 🔥 关键修复：启用默认的对话历史处理
        output_key="answer"
    )

    st.success("✅ 中医诊疗问答模型初始化完成（带上下文记忆）！")
    return qa_chain


# ===================== Streamlit主界面 =====================
def main():
    st.set_page_config(
        page_title="中医RAG助手（硅基流动嵌入版）",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🏥 中医临床诊疗RAG助手")
    st.subheader("✨ 基于硅基流动 (SiliconFlow) | 增强上下文记忆")
    st.divider()

    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 硅基流动配置")

        # API Key配置
        api_key = st.text_input(
            "硅基流动API Key",
            value=Config.SILICONFLOW_EMBEDDING_CONFIG["api_key"],
            type="password",
            help="从硅基流动控制台获取：https://siliconflow.cn"
        )
        if api_key:
            Config.SILICONFLOW_EMBEDDING_CONFIG["api_key"] = api_key
            Config.SILICONFLOW_LLM_CONFIG["api_key"] = api_key

        # 嵌入模型选择
        embedding_model = st.selectbox(
            "选择嵌入模型",
            Config.EMBEDDING_MODELS,
            index=0,
            help="硅基流动托管的嵌入模型"
        )

        # LLM模型选择
        llm_model = st.selectbox(
            "选择LLM模型",
            ["Qwen/Qwen2-7B-Instruct", "Qwen/Qwen2-14B-Instruct", "Meta-Llama-3-8B-Instruct"],
            index=0
        )
        Config.SILICONFLOW_LLM_CONFIG["model"] = llm_model

        # 上下文记忆配置
        st.markdown("---")
        st.header("🧠 上下文记忆配置")
        max_history = st.slider(
            "最大记忆轮数",
            min_value=3,
            max_value=20,
            value=Config.CONTEXT_MEMORY_CONFIG["max_history_length"],
            help="控制对话历史的记忆长度，过大会增加Token消耗"
        )
        Config.CONTEXT_MEMORY_CONFIG["max_history_length"] = max_history

        # 文档路径配置
        doc_path = st.text_input(
            "文档路径",
            value=str(Config.DOC_PATH),
            help="中医诊疗文档路径（txt格式）"
        )
        Config.DOC_PATH = Path(doc_path)

        # 重置按钮
        col1, col2, col3 = st.columns(3)
        with col1:
            reset_vector_db = st.button("🔄 重置向量库", type="secondary")
        with col2:
            if st.button("♻️ 清空缓存", type="secondary"):
                st.cache_resource.clear()
                st.rerun()
        with col3:
            if st.button("🧹 清空记忆", type="secondary"):
                st.session_state.chat_history = []
                if "similar_docs_history" in st.session_state:
                    del st.session_state["similar_docs_history"]
                st.success("已清空对话记忆！")
                st.rerun()

        # 使用说明
        st.markdown("---")
        st.info(
            "📝 使用说明：\n"
            "1. 输入硅基流动API Key\n"
            "2. 选择嵌入模型和LLM模型\n"
            "3. 调整上下文记忆轮数\n"
            "4. 确认文档路径正确\n"
            "5. 在对话框中输入中医问题\n"
            "6. 系统会记忆对话历史，提供连贯回答"
        )

        # 注意事项
        st.markdown("---")
        st.warning(
            "⚠️ 注意事项：\n"
            "1. 切换嵌入模型会自动重置向量库\n"
            "2. 模型调用会产生相应的计费\n"
            "3. 确保API Key有对应模型的权限\n"
            "4. 记忆轮数越多，Token消耗越大"
        )

    # 初始化会话状态
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # 格式：List[Tuple[str, str]]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None
    if "similar_docs_history" not in st.session_state:
        st.session_state.similar_docs_history = []

    # 系统初始化流程
    with st.spinner("系统初始化中..."):
        # 1. 初始化嵌入模型
        if not st.session_state.embeddings:
            st.session_state.embeddings = init_siliconflow_embeddings(embedding_model)

        # 2. 加载文档
        raw_docs = load_documents(Config.DOC_PATH)

        # 3. 切割文本
        if raw_docs and st.session_state.embeddings:
            split_docs = split_documents(raw_docs)

            # 4. 创建向量库
            if split_docs and (reset_vector_db or not st.session_state.vector_store):
                st.session_state.vector_store = create_vector_store(
                    split_docs,
                    st.session_state.embeddings,
                    reset=reset_vector_db
                )

            # 5. 创建问答链
            if st.session_state.vector_store and not st.session_state.qa_chain:
                st.session_state.qa_chain = create_qa_chain(st.session_state.vector_store)

    # 上下文记忆展示
    col1, col2 = st.columns([7, 3])

    with col2:
        st.header("📜 对话记忆")
        if st.session_state.chat_history:
            st.info(
                f"当前记忆轮数：{len(st.session_state.chat_history)}/{Config.CONTEXT_MEMORY_CONFIG['max_history_length']}")

            # 折叠显示完整对话历史
            with st.expander("查看完整对话记忆", expanded=False):
                formatted_history = format_chat_history_for_display(st.session_state.chat_history)
                st.text_area(
                    "对话历史",
                    value=formatted_history,
                    height=300,
                    disabled=True
                )
        else:
            st.info("暂无对话记忆，请开始提问...")

    with col1:
        # 聊天界面
        st.subheader("💬 对话界面")

        # 显示历史对话
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.chat_message("user", avatar="👤"):
                st.markdown(question)
            with st.chat_message("assistant", avatar="🏥"):
                st.markdown(answer)

                # 显示上下文关联提示
                if i > 0:
                    st.caption(f"💡 关联上文：第{i}轮对话")

                # 显示参考文档
                if i < len(st.session_state.get("similar_docs_history", [])):
                    similar_docs = st.session_state["similar_docs_history"][i]
                    if similar_docs:
                        with st.expander(f"📊 参考文档（相似度排序）", expanded=False):
                            for j, (doc, score) in enumerate(similar_docs, 1):
                                st.markdown(f"### 参考文档 {j}（相似度：{score:.4f}）")
                                st.markdown(f"**来源**：{doc.metadata.get('file_name', '未知')}")
                                st.markdown(f"**编码**：{doc.metadata.get('encoding', '未知')}")
                                st.markdown(f"**内容**：{doc.page_content[:800]}..." if len(
                                    doc.page_content) > 800 else doc.page_content)
                                st.divider()

        # 聊天输入
        if prompt := st.chat_input("请输入你的中医诊疗问题..."):
            if not st.session_state.qa_chain:
                st.error("❌ 问答模型尚未初始化完成，请检查配置！")
            else:
                with st.chat_message("user", avatar="👤"):
                    st.markdown(prompt)

                with st.chat_message("assistant", avatar="🏥"):
                    with st.spinner("🔍 正在检索并生成回答（结合上下文记忆）..."):
                        try:
                            # 获取相似度结果
                            similar_docs = get_similar_documents(st.session_state.vector_store, prompt,
                                                                 k=Config.RETRIEVER_CONFIG["k"])

                            # 🔥 核心修复：传递正确格式的对话历史（List[Tuple[str, str]]）
                            # ConversationalRetrievalChain 要求 chat_history 是列表元组格式
                            result = st.session_state.qa_chain({
                                "question": prompt,
                                "chat_history": st.session_state.chat_history  # 直接传递原始格式
                            })

                            # 显示回答
                            answer = result["answer"].strip()
                            st.markdown(answer)

                            # 保存对话历史（保持List[Tuple[str, str]]格式）
                            st.session_state.chat_history.append((prompt, answer))
                            st.session_state.similar_docs_history.append(similar_docs)

                            # 限制历史长度
                            if len(st.session_state.chat_history) > Config.CONTEXT_MEMORY_CONFIG["max_history_length"]:
                                st.session_state.chat_history = st.session_state.chat_history[
                                                                -Config.CONTEXT_MEMORY_CONFIG["max_history_length"]:]
                                st.session_state.similar_docs_history = st.session_state.similar_docs_history[
                                                                        -Config.CONTEXT_MEMORY_CONFIG[
                                                                            "max_history_length"]:]

                            # 显示参考文档
                            with st.expander("📚 参考文档（按相似度排序）", expanded=False):
                                st.info(f"📝 基于 {len(similar_docs)} 个相关文档生成回答")
                                for i, (doc, score) in enumerate(similar_docs, 1):
                                    st.markdown(f"### 参考文档 {i}（相似度：{score:.4f}）")
                                    st.markdown(f"**来源文件**：{doc.metadata.get('file_name', '未知')}")
                                    st.markdown(f"**文件编码**：{doc.metadata.get('encoding', '未知')}")
                                    st.markdown(f"**文件路径**：{doc.metadata.get('source', '未知')}")
                                    st.markdown(f"**内容**：{doc.page_content[:800]}..." if len(
                                        doc.page_content) > 800 else doc.page_content)
                                    st.divider()

                            # 显示上下文关联提示
                            if len(st.session_state.chat_history) > 1:
                                st.caption("💡 回答已结合之前的对话上下文，保持诊疗建议的连贯性")

                        except Exception as e:
                            st.error(f"❌ 回答生成失败：{str(e)}")
                            logger.error(f"问答出错：{str(e)}", exc_info=True)

        # 清空对话按钮
        col_clear1, col_clear2 = st.columns([1, 9])
        with col_clear1:
            if st.button("🗑️ 清空对话", type="primary"):
                st.session_state.chat_history = []
                st.session_state.similar_docs_history = []
                st.rerun()


if __name__ == "__main__":
    main()