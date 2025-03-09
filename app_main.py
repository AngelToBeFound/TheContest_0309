import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import random
import time
import pandas as pd
import matplotlib.pyplot as plt

# 设置页面配置
st.set_page_config(page_title="数学问题生成与批改系统", layout="wide")

# 初始化会话状态
if "questions" not in st.session_state:
    st.session_state["questions"] = []
if "user_answers" not in st.session_state:
    st.session_state["user_answers"] = {}
if "feedback" not in st.session_state:
    st.session_state["feedback"] = {}
if "history" not in st.session_state:
    st.session_state["history"] = []

# 加载Hugging Face模型
@st.cache_resource(show_spinner=True)  # 修正参数为 show_spinner=True
def load_huggingface_model():
    st.info("正在从Hugging Face加载模型，请稍候...")
    try:
        # 替换为您自己的模型，例如 "your-username/deepseek-math-assistant"
        model_name = "distilbert-base-uncased"  # 示例模型，可替换
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)
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
        response = generator(prompt, max_length=100, temperature=0.7, num_return_sequences=1)[0]["generated_text"]
        # 模拟解析生成的内容（因distilbert非最佳生成模型，实际替换为DeepSeek更佳）
        if topic == "线性方程":
            a, b = random.randint(1, 5 * difficulty), random.randint(1, 5 * difficulty)
            correct_answer = (a * difficulty + b - b) // a  # 简化计算
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
    
    prompt = f"批改答案，题目：{question}，用户答案：{user_answer}"
    try:
        feedback = generator(prompt, max_length=50, temperature=0.7, num_return_sequences=1)[0]["generated_text"]
        # 简单逻辑模拟（实际应基于模型训练结果）
        if "待AI解答" in user_answer:
            return "AI正在计算中，请稍候..."
        return feedback
    except Exception as e:
        return f"批改失败：{str(e)}"

# 主界面
st.title("数学问题生成与批改系统")
st.markdown("这是一个基于Hugging Face模型的数学助教工具，支持生成题目、批改答案和学习统计。")

# 侧边栏：设置选项
st.sidebar.header("题目设置")
topic = st.sidebar.selectbox("选择数学主题", ["线性方程", "二次方程", "分数运算"])
difficulty = st.sidebar.slider("难度等级", 1, 5, 3, help="1为最简单，5为最难")
num_questions = st.sidebar.number_input("生成题目数量", 1, 5, 2)
generate_button = st.sidebar.button("生成题目")

# 生成题目
if generate_button:
    with st.spinner("正在生成题目，请稍候..."):
        st.session_state["questions"] = []
        for i in range(num_questions):
            question_data = generate_math_question(topic, difficulty)
            question_data["id"] = i
            st.session_state["questions"].append(question_data)
        time.sleep(1)  # 模拟加载时间
        st.success(f"成功生成 {num_questions} 道题目！")

# 显示题目和交互
if st.session_state["questions"]:
    st.subheader("题目列表")
    tabs = st.tabs([f"题目 {i+1}" for i in range(len(st.session_state["questions"]))])
    
    for idx, tab in enumerate(tabs):
        with tab:
            q = st.session_state["questions"][idx]
            st.write(f"**题目 {q['id'] + 1}**: {q['text']}")
            st.write("选项：", ", ".join(map(str, q["options"])))
            
            # 用户选择答案
            answer_key = f"answer_{q['id']}"
            selected_answer = st.selectbox("选择你的答案", q["options"], key=answer_key)
            st.session_state["user_answers"][q["id"]] = selected_answer
            
            # 提交按钮
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

# 学习统计
st.header("学习统计")
if st.session_state["history"]:
    df = pd.DataFrame(st.session_state["history"])
    st.write("### 答题历史")
    st.dataframe(df)
    
    # 统计正确率
    correct_count = sum(1 for h in st.session_state["history"] if "正确" in h["feedback"])
    total = len(st.session_state["history"])
    st.write(f"正确率：{correct_count / total * 100:.2f}%")
    
    # 可视化
    fig, ax = plt.subplots()
    ax.pie([correct_count, total - correct_count], labels=["正确", "错误"], autopct="%1.1f%%", colors=["#66b3ff", "#ff9999"])
    st.pyplot(fig)

# 侧边栏：额外功能
st.sidebar.markdown("---")
st.sidebar.header("其他功能")
if st.sidebar.button("清空历史"):
    st.session_state["questions"] = []
    st.session_state["user_answers"] = {}
    st.session_state["feedback"] = {}
    st.session_state["history"] = []
    st.sidebar.success("历史记录已清空！")

st.sidebar.info("选择主题和难度生成题目，提交答案后查看AI反馈，统计答题表现。")