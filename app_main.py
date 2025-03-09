import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# 设置页面配置
st.set_page_config(page_title="AI对话助手", layout="wide")

# 初始化会话状态
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# 加载Hugging Face模型
@st.cache_resource(show_spinner=True)
def load_huggingface_model():
    st.info("正在从Hugging Face加载模型，请稍候...")
    try:
        model_name = "gpt2"  # 替换为 "your-username/deepseek-math-assistant"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        st.success("模型加载成功！")
        return generator
    except Exception as e:
        st.error(f"模型加载失败：{str(e)}")
        return None

generator = load_huggingface_model()

# AI对话函数
def chat_with_ai(user_input):
    if not generator:
        return "模型未加载，无法对话"
    
    prompt = f"用户提问：{user_input}\nAI回答："
    try:
        response = generator(prompt, max_length=100, temperature=0.7, num_return_sequences=1)[0]["generated_text"]
        return response.split("AI回答：")[-1].strip()
    except Exception as e:
        return f"对话失败：{str(e)}"

# 主界面
st.title("AI对话助手")
st.markdown("与AI进行实时对话，输入您的问题或请求，获取智能回答！")

# 对话输入和显示
user_input = st.text_area("输入您的问题或请求", height=100)
if st.button("发送"):
    if user_input:
        with st.spinner("AI思考中..."):
            response = chat_with_ai(user_input)
            st.session_state["chat_history"].append({"user": user_input, "ai": response})
        st.success("对话已更新！")
    else:
        st.warning("请输入内容后再发送！")

# 显示聊天历史
if st.session_state["chat_history"]:
    st.subheader("对话历史")
    for chat in st.session_state["chat_history"]:
        st.markdown(f"**您**: {chat['user']}")
        st.markdown(f"**AI**: {chat['ai']}")
        st.markdown("---")

# 侧边栏：清空历史
st.sidebar.header("工具")
if st.sidebar.button("清空对话历史"):
    st.session_state["chat_history"] = []
    st.sidebar.success("对话历史已清空！")

st.sidebar.info("输入问题与AI对话，点击‘发送’获取回答，侧边栏可清空历史。")