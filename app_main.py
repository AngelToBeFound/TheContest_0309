import streamlit as st
import random

# 界面标题
st.title("随机数学题目生成器")

# 用户选择难度
st.header("生成题目")
difficulty = st.selectbox("选择难度", ["简单", "中等"])

# 生成题目按钮
if st.button("生成题目"):
    if difficulty == "简单":
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        correct_answer = a + b
        question = f"{a} + {b} = ?"
        options = [correct_answer, correct_answer + 1, correct_answer - 1, correct_answer + 2]
    else:  # 中等
        a = random.randint(10, 50)
        b = random.randint(10, 50)
        correct_answer = a - b
        question = f"{a} - {b} = ?"
        options = [correct_answer, correct_answer + 1, correct_answer - 1, correct_answer + 2]

    random.shuffle(options)  # 打乱选项顺序
    st.session_state["question"] = question
    st.session_state["options"] = options
    st.session_state["correct_answer"] = correct_answer
    st.write(f"题目：{question}")
    st.write("选项：", ", ".join(map(str, options)))

# 用户输入答案
st.header("提交答案")
user_answer = st.text_input("请输入你的答案（数字）", "")

# 检查答案按钮
if st.button("检查答案"):
    if "correct_answer" not in st.session_state:
        st.error("请先生成题目！")
    else:
        try:
            user_answer = int(user_answer)
            if user_answer == st.session_state["correct_answer"]:
                st.success("回答正确！")
            else:
                st.error(f"回答错误，正确答案是 {st.session_state['correct_answer']}。")
        except ValueError:
            st.error("请输入有效的数字！")

# 添加说明
st.info("这是一个简单的数学题目生成器，选择难度后点击‘生成题目’，然后输入答案检查对错。")