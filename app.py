"""Streamlit主应用"""
import sys
print("Python解释器路径：", sys.executable)
print("模块搜索路径：", sys.path[:3])  # 只看前3个关键路径
import streamlit as st
from pathlib import Path
from config.settings import Config
from core.document_loader import load_documents, split_documents
from core.embedding import init_siliconflow_embeddings
from core.vector_store import create_vector_store
from core.qa_chain import create_qa_chain
from core.similarity import get_similar_documents, is_irrelevant_question
from utils.logger import logger
from utils.chat_history import format_chat_history_for_display, trim_chat_history

def main():
    # 页面配置
    st.set_page_config(
        page_title="中医RAG助手（硅基流动嵌入版）",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 页面标题
    st.title("🏥 中医临床诊疗RAG助手")
    st.subheader("✨ 基于硅基流动 (SiliconFlow) | 增强上下文记忆")
    st.divider()

    # ===================== 侧边栏配置 =====================
    with st.sidebar:
        st.header("⚙️ 系统配置")

        # API Key配置
        api_key = st.text_input(
            "硅基流动API Key",
            value=Config.SILICONFLOW_CONFIG["api_key"],
            type="password",
            help="从硅基流动控制台获取：https://siliconflow.cn"
        )
        if api_key:
            Config.SILICONFLOW_CONFIG["api_key"] = api_key

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
            Config.LLM_MODELS,
            index=0
        )
        Config.SILICONFLOW_CONFIG["llm_model"] = llm_model

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

        # 无关问题阈值
        st.markdown("---")
        st.header("🚫 无关问题配置")
        irrelevant_threshold = st.slider(
            "相似度阈值（判定无关）",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="低于此值的问题判定为无关问题"
        )

        # 文档路径配置
        st.markdown("---")
        st.header("📄 文档配置")
        doc_path = st.text_input(
            "文档路径",
            value=str(Config.DOC_PATH),
            help="中医诊疗文档路径（txt格式）"
        )
        Config.DOC_PATH = Path(doc_path)

        # 功能按钮
        st.markdown("---")
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
                st.session_state.similar_docs_history = []
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
            "5. 在对话框中输入中医问题"
        )

    # ===================== 会话状态初始化 =====================
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None
    if "similar_docs_history" not in st.session_state:
        st.session_state.similar_docs_history = []

    # ===================== 系统初始化 =====================
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

    # ===================== 界面布局 =====================
    col1, col2 = st.columns([7, 3])

    # 右侧：对话记忆展示
    with col2:
        st.header("📜 对话记忆")
        if st.session_state.chat_history:
            st.info(
                f"当前记忆轮数：{len(st.session_state.chat_history)}/{Config.CONTEXT_MEMORY_CONFIG['max_history_length']}")

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

    # 左侧：聊天界面
    with col1:
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
                                content = doc.page_content
                                st.markdown(f"**内容**：{content[:800]}..." if len(content) > 800 else content)
                                st.divider()

        # 聊天输入
        if prompt := st.chat_input("请输入你的中医诊疗问题..."):
            if not st.session_state.qa_chain:
                st.error("❌ 问答模型尚未初始化完成，请检查配置！")
            else:
                with st.chat_message("user", avatar="👤"):
                    st.markdown(prompt)

                with st.chat_message("assistant", avatar="🏥"):
                    with st.spinner("🔍 正在检索并生成回答..."):
                        try:
                            # 判断是否为无关问题
                            irrelevant = is_irrelevant_question(
                                st.session_state.vector_store,
                                prompt,
                                irrelevant_threshold
                            )

                            if irrelevant:
                                # 无关问题固定回复
                                answer = Config.IRRELEVANT_REPLY
                                st.markdown(answer)
                                st.caption("⚠️ 该问题与中医临床诊疗无关")
                                similar_docs = []
                            else:
                                # 相关问题：调用问答链
                                result = st.session_state.qa_chain({
                                    "question": prompt,
                                    "chat_history": st.session_state.chat_history
                                })
                                answer = result["answer"].strip()
                                st.markdown(answer)
                                similar_docs = get_similar_documents(st.session_state.vector_store, prompt)

                                # 显示参考文档
                                with st.expander("📚 参考文档（按相似度排序）", expanded=False):
                                    st.info(f"📝 基于 {len(similar_docs)} 个相关文档生成回答")
                                    for i, (doc, score) in enumerate(similar_docs, 1):
                                        st.markdown(f"### 参考文档 {i}（相似度：{score:.4f}）")
                                        st.markdown(f"**来源文件**：{doc.metadata.get('file_name', '未知')}")
                                        st.markdown(f"**内容**：{doc.page_content[:800]}..." if len(doc.page_content) > 800 else doc.page_content)
                                        st.divider()

                            # 保存对话历史
                            st.session_state.chat_history.append((prompt, answer))
                            st.session_state.similar_docs_history.append(similar_docs)

                            # 裁剪对话历史到最大长度
                            st.session_state.chat_history = trim_chat_history(st.session_state.chat_history)
                            st.session_state.similar_docs_history = st.session_state.similar_docs_history[-Config.CONTEXT_MEMORY_CONFIG["max_history_length"]:]

                            # 上下文关联提示
                            if len(st.session_state.chat_history) > 1:
                                st.caption("💡 回答已结合之前的对话上下文")

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