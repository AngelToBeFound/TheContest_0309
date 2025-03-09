import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import random
import time
import pandas as pd
import matplotlib.pyplot as plt

# 设置页面配置
st.set_page_config(page_title="高级数学AI助教", layout="wide")

# 初始化会话状态
if "questions" not in st.session_state:
    st.session_state["questions"] = []
if "user_answers" not in st.session_state:
    st.session_state["user_answers"] = {}
if "feedback" not in st.session_state:
    st.session_state["feedback"] = {}
if "history" not in st.session_state:
    st.session_state["history"] = []
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

# 生成数学题目函数
def generate_math_question(topic, difficulty):
    if not generator:
        return {"text": "模型未加载，无法生成题目", "options": [], "answer": ""}
    
    prompt = f"生成一道{topic}选择题，难度等级{difficulty}，格式：题目+选项+答案"
    try:
        if topic == "线性方程":
            a, b = random.randint(1, 5 * difficulty), random.randint(1, 5 * difficulty)
            correct_answer = (a * difficulty + b - b) // a
            question = f"{a}x + {b} = {a * difficulty + b}, x = ?"
            options = [correct_answer, correct_answer + 1, correct_answer - 1, correct_answer + 2]
        elif topic == "二次方程":
            question = f"x^2 + {difficulty}x + {difficulty} = 0, 求x"
            options = ["待AI解答", "无解", "1", "-1"]
            correct_answer = "待AI解答"
        else:  # 分数运算
            question = f"1/{difficulty} + 1/{difficulty + 1} = ?"
            options = ["待AI解答", "1", "2", "0"]
            correct_answer = "待AI解答"
        
        random.shuffle(options)
        return {"text": question, "options": options, "answer": str(correct_answer)}
    except Exception as e:
        return {"text": f"生成失败：{str(e)}", "options": [], "answer": ""}

# 批改答案函数
def grade_answer(question, user_answer):
    if not generator:
        return "模型未加载，无法批改"
    
    user_answer_str = str(user_answer)
    prompt = f"批改答案，题目：{question}，用户答案：{user_answer_str}"
    try:
        feedback = generator(prompt, max_length=50, temperature=0.7, num_return_sequences=1)[0]["generated_text"]
        if "待AI解答" in user_answer_str:
            return "AI正在计算中，请稍候..."
        return feedback
    except Exception as e:
        return f"批改失败：{str(e)}"

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

# 一键教案生成函数
def generate_lesson_plan(topic, difficulty):
    if not generator:
        return "模型未加载，无法生成教案"
    
    prompt = f"生成一份关于{topic}的教案，难度等级{difficulty}，包含教学目标、3道例题及其答案"
    try:
        plan = generator(prompt, max_length=300, temperature=0.8, num_return_sequences=1)[0]["generated_text"]
        return plan
    except Exception as e:
        return f"教案生成失败：{str(e)}"

# 主界面
st.title("高级数学AI助教")
st.markdown("这是一个多功能的数学助教工具，支持题目生成、批改、学习统计、AI对话和教案生成。")

# 多列布局
col1, col2 = st.columns([2, 1])

# 左侧：题目生成与批改
with col1:
    st.header("题目生成与批改")
    topic = st.selectbox("选择数学主题", ["线性方程", "二次方程", "分数运算"])
    difficulty = st.slider("难度等级", 1, 5, 3, help="1为最简单，5为最难")
    num_questions = st.number_input("生成题目数量", 1, 5, 2)
    if st.button("生成题目"):
        with st.spinner("正在生成题目..."):
            st.session_state["questions"] = []
            for i in range(num_questions):
                question_data = generate_math_question(topic, difficulty)
                question_data["id"] = i
                st.session_state["questions"].append(question_data)
            st.success(f"成功生成 {num_questions} 道题目！")

    if st.session_state["questions"]:
        st.subheader("题目列表")
        tabs = st.tabs([f"题目 {i+1}" for i in range(len(st.session_state["questions"]))])
        for idx, tab in enumerate(tabs):
            with tab:
                q = st.session_state["questions"][idx]
                st.write(f"**题目 {q['id'] + 1}**: {q['text']}")
                st.write("选项：", ", ".join(map(str, q["options"])))
                answer_key = f"answer_{q['id']}"
                selected_answer = st.selectbox("选择你的答案", q["options"], key=answer_key)
                st.session_state["user_answers"][q["id"]] = selected_answer
                if st.button("提交答案", key=f"submit_{q['id']}"):
                    feedback = grade_answer(q["text"], selected_answer)
                    st.session_state["feedback"][q["id"]] = feedback
                    st.session_state["history"].append({
                        "question": q["text"],
                        "answer": selected_answer,
                        "feedback": feedback,
                        "time": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    st.write(f"**反馈**: {feedback}")

# 右侧：AI对话框和教案生成
with col2:
    st.header("AI对话框")
    user_input = st.text_area("输入您的问题或请求", height=100)
    if st.button("发送"):
        with st.spinner("AI思考中..."):
            response = chat_with_ai(user_input)
            st.session_state["chat_history"].append({"user": user_input, "ai": response})
    if st.session_state["chat_history"]:
        for chat in st.session_state["chat_history"]:
            st.markdown(f"**您**: {chat['user']}")
            st.markdown(f"**AI**: {chat['ai']}")
            st.markdown("---")

    st.header("一键教案生成")
    if st.button("生成教案"):
        with st.spinner("生成教案中..."):
            lesson_plan = generate_lesson_plan(topic, difficulty)
            st.text_area("教案内容", lesson_plan, height=200)

# 学习统计
st.header("学习统计")
if st.session_state["history"]:
    df = pd.DataFrame(st.session_state["history"])
    st.write("### 答题历史")
    st.dataframe(df)
    correct_count = sum(1 for h in st.session_state["history"] if "正确" in h["feedback"])
    total = len(st.session_state["history"])
    st.write(f"正确率：{correct_count / total * 100:.2f}%")
    fig, ax = plt.subplots()
    ax.pie([correct_count, total - correct_count], labels=["正确", "错误"], autopct="%1.1f%%", colors=["#66b3ff", "#ff9999"])
    st.pyplot(fig)

# 侧边栏：额外功能
st.sidebar.header("工具箱")
if st.sidebar.button("清空所有记录"):
    st.session_state["questions"] = []
    st.session_state["user_answers"] = {}
    st.session_state["feedback"] = {}
    st.session_state["history"] = []
    st.session_state["chat_history"] = []
    st.sidebar.success("所有记录已清空！")

st.sidebar.info("使用左侧生成题目并批改，右侧与AI对话或生成教案，下方查看学习统计。")