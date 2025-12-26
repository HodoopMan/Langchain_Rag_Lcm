# 中医临床诊疗RAG助手

基于LangChain + Streamlit + 硅基流动API构建的中医RAG问答系统，支持上下文记忆、相似性检索、无关问题过滤。

## 快速开始

### 1. 环境准备
```bash
# 克隆项目
git clone <your-repo-url>
cd llm-tcm-rag

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt