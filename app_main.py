import streamlit as st
import google.generativeai as genai
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Message:
    role: str
    content: str

class ChatState:
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
        self.model: Optional[genai.GenerativeModel] = None
        self.api_key_valid: bool = False

    def clear_history(self):
        self.messages = []

    def add_message(self, user_input: str, ai_response: str):
        self.messages.append({"user": user_input, "ai": ai_response})

class GeminiChat:
    def __init__(self):
        self.model = None

    def initialize_model(self, api_key: str) -> bool:
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            return False

    def get_response(self, user_input: str, history: List[Dict[str, str]]) -> str:
        if not self.model:
            return "模型未初始化，请检查API密钥。"

        try:
            formatted_history = []
            for chat in history:
                formatted_history.extend([
                    {"role": "user", "parts": [chat["user"]]},
                    {"role": "model", "parts": [chat["ai"]]}
                ])
            
            chat = self.model.start_chat(history=formatted_history)
            response = chat.send_message(user_input)
            return response.text.strip()
        except Exception as e:
            error_msg = f"对话生成失败: {str(e)}"
            logger.error(error_msg)
            return error_msg

class ChatUI:
    def __init__(self):
        self.state = self._initialize_state()
        self.gemini_chat = GeminiChat()
        self._setup_page()
        self._setup_sidebar()
        self._create_main_layout()

    @staticmethod
    def _initialize_state() -> ChatState:
        if "chat_state" not in st.session_state:
            st.session_state.chat_state = ChatState()
        return st.session_state.chat_state

    def _setup_page(self):
        st.set_page_config(
            page_title="Gemini 智能助手",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.title("Gemini 智能助手")
        st.markdown("💡 基于 Google Gemini API 的新一代AI对话助手")

    def _setup_sidebar(self):
        st.sidebar.header("⚙️ 设置")
        api_key = st.sidebar.text_input(
            "Gemini API 密钥",
            type="password",
            help="请输入您的 Gemini API 密钥以开始对话"
        )

        if api_key and not self.state.api_key_valid:
            with st.sidebar.spinner("正在验证API密钥..."):
                if self.gemini_chat.initialize_model(api_key):
                    self.state.api_key_valid = True
                    self.state.model = self.gemini_chat.model
                    st.sidebar.success("✅ API密钥验证成功！")
                else:
                    st.sidebar.error("❌ API密钥无效或验证失败")

        if st.sidebar.button("🗑️ 清空对话历史", help="点击清除所有对话记录"):
            self.state.clear_history()
            st.sidebar.success("已清空对话历史！")
            st.experimental_rerun()

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 💡 使用说明")
        st.sidebar.markdown("""
        1. 输入 Gemini API 密钥
        2. 在右侧输入框中输入问题
        3. 点击发送或按回车键
        4. AI 回答将显示在左侧
        """)

    def _create_main_layout(self):
        col1, col2 = st.columns([1, 1])

        # AI回复区域
        with col1:
            st.subheader("🤖 AI 回复")
            self._display_ai_messages()

        # 用户输入区域
        with col2:
            st.subheader("👤 用户输入")
            self._handle_user_input()
            self._display_user_messages()

    def _display_ai_messages(self):
        for msg in self.state.messages:
            with st.chat_message("assistant"):
                st.markdown(msg["ai"])

    def _display_user_messages(self):
        for msg in self.state.messages:
            with st.chat_message("user"):
                st.markdown(msg["user"])

    def _handle_user_input(self):
        if not self.state.api_key_valid:
            st.warning("⚠️ 请先在侧边栏输入有效的 Gemini API 密钥")
            return

        user_input = st.text_area(
            "输入您的问题",
            height=100,
            placeholder="在这里输入您的问题..."
        )

        if st.button("发送", type="primary"):
            if not user_input:
                st.warning("请输入内容后再发送！")
                return

            with st.spinner("🤔 AI思考中..."):
                response = self.gemini_chat.get_response(
                    user_input,
                    self.state.messages
                )
                self.state.add_message(user_input, response)
            st.experimental_rerun()

def main():
    chat_ui = ChatUI()

if __name__ == "__main__":
    main()