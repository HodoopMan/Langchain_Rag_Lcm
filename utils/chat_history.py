"""对话历史处理工具"""
from typing import List, Tuple
from config.settings import Config

def format_chat_history_for_display(chat_history: List[Tuple[str, str]]) -> str:
    """格式化对话历史用于前端展示"""
    if not chat_history:
        return "无"

    # 限制历史长度
    history_to_use = chat_history[-Config.CONTEXT_MEMORY_CONFIG["max_history_length"]:]

    formatted_history = []
    for i, (human, ai) in enumerate(history_to_use, 1):
        formatted_history.append(f"用户{i}：{human}")
        formatted_history.append(f"助手{i}：{ai}")

    return Config.CONTEXT_MEMORY_CONFIG["history_separator"].join(formatted_history)

def trim_chat_history(chat_history: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """裁剪对话历史到最大长度"""
    max_len = Config.CONTEXT_MEMORY_CONFIG["max_history_length"]
    if len(chat_history) > max_len:
        return chat_history[-max_len:]
    return chat_history