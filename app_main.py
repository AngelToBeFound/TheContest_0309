import streamlit as st
import google.generativeai as genai

# 设置页面配置
st.set_page_config(page_title="Gemini 对话助手", layout="wide")

# 初始化会话状态
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "gemini_model" not in st.session_state:
    st.session_state["gemini_model"] = None
if "api_key_valid" not in st.session_state:
    st.session_state["api_key_valid"] = False

# 侧边栏：输入 API 密钥
st.sidebar.header("设置")
api_key = st.sidebar.text_input("输入您的 Gemini API 密钥", type="password", key="api_key_input")

# 初始化 Gemini 模型
def initialize_gemini_model(api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        st.session_state["gemini_model"] = model
        st.session_state["api_key_valid"] = True
        st.sidebar.success("API 密钥有效，模型已加载！")
    except Exception as e:
        st.session_state["api_key_valid"] = False
        st.session_state["gemini_model"] = None
        st.sidebar.error(f"API 密钥无效或加载失败：{str(e)}")

# 检查并初始化模型
if api_key and not st.session_state["api_key_valid"]:
    initialize_gemini_model(api_key)

# AI 对话函数
def chat_with_gemini(user_input):
    model = st.session_state["gemini_model"]
    if not model:
        return "模型未加载，请输入有效的 API 密钥。"
    
    # 修复后的对话历史格式
    history = [
        {"role": "user", "parts": [chat["user"]]},
        {"role": "model", "parts": [chat["ai"]]}
        for chat in st.session_state["chat_history"]
    ]
    # 将历史展平为连续的列表
    flat_history = [item for sublist in history for item in sublist]
    try:
        chat = model.start_chat(history=flat_history)
        response = chat.send_message(user_input)
        return response.text.strip()
    except Exception as e:
        return f"对话失败：{str(e)}"

# 主界面
st.title("Gemini 对话助手")
st.markdown("基于 Google Gemini API 的智能对话，AI 在左侧，用户在右侧。")

# 创建左右两列
col1, col2 = st.columns([1, 1])

# 左侧：AI 回答
with col1:
    st.subheader("AI")
    if st.session_state["chat_history"]:
        for chat in st.session_state["chat_history"]:
            st.markdown(f"**AI**: {chat['ai']}")
            st.markdown("---")

# 右侧：用户输入和历史
with col2:
    st.subheader("您")
    if not st.session_state["api_key_valid"]:
        st.warning("请在侧边栏输入有效的 Gemini API 密钥以开始对话。")
    else:
        user_input = st.text_area("输入您的问题或请求", height=100, key="user_input")
        if st.button("发送"):
            if user_input:
                with st.spinner("Gemini 思考中..."):
                    response = chat_with_gemini(user_input)
                    st.session_state["chat_history"].append({"user": user_input, "ai": response})
                st.experimental_rerun()
            else:
                st.warning("请输入内容后再发送！")
    
    if st.session_state["chat_history"]:
        for chat in st.session_state["chat_history"]:
            st.markdown(f"**您**: {chat['user']}")
            st.markdown("---")

# 侧边栏：清空历史
st.sidebar.header("工具")
if st.sidebar.button("清空对话历史"):
    st.session_state["chat_history"] = []
    st.sidebar.success("对话历史已清空！")
    st.experimental_rerun()

st.sidebar.info("在侧边栏输入 Gemini API 密钥后，右侧输入问题，AI 将在左侧回答。")