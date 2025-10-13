# 三人对话：求职面试场景
import os
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.agents import ChatAgent
from camel.messages import BaseMessage

from dotenv import load_dotenv
load_dotenv()  # 自动加载 .env 文件

BASE_URL = os.getenv("QDD_BASE_URL")
API_KEY  = os.getenv("QDD_API_KEY")
MODEL_ID = os.getenv("QDD_MODEL",    "gpt-4o")

model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type=MODEL_ID,
    api_key=API_KEY,
    url=BASE_URL,
    model_config_dict={
        "temperature": 0.7,
        "max_tokens": 1200,
    },
)

# ==== 创建技术面试官 Agent ====
interviewer_system_message = BaseMessage.make_assistant_message(
    role_name="Technical Interviewer",
    content=(
        "你是一名技术面试官，负责评估候选人的技术能力。你的职责是：\n"
        "1. 提出有深度的技术问题\n"
        "2. 评估候选人的回答质量\n"
        "3. 适时追问以了解候选人的真实水平\n"
        "4. 与HR配合完成面试\n"
        "5. 保持专业但友好的态度\n\n"
        "请严格使用以下格式输出：\n"
        "[INTERVIEWER]\n"
        "提问/评价: <技术问题或对候选人回答的评价>\n"
        "观察点: <候选人的表现观察>\n"
        "后续动作: <接下来要做什么>\n"
    )
)

interviewer_agent = ChatAgent(
    system_message=interviewer_system_message,
    model=model,
    message_window_size=25,
    token_limit=8192,
)

# ==== 创建HR Agent ====
hr_system_message = BaseMessage.make_assistant_message(
    role_name="HR",
    content=(
        "你是HR，负责协调面试流程和评估候选人综合素质。你的职责是：\n"
        "1. 介绍面试流程和公司情况\n"
        "2. 询问候选人的职业规划和期望\n"
        "3. 补充技术面试官未涉及的软技能问题\n"
        "4. 关注候选人的沟通能力和文化匹配度\n"
        "5. 在适当时候总结面试\n\n"
        "请严格使用以下格式输出：\n"
        "[HR]\n"
        "沟通内容: <询问的问题或说明的信息>\n"
        "关注点: <对候选人的观察>\n"
        "建议: <给技术面试官或候选人的建议>\n"
    )
)

hr_agent = ChatAgent(
    system_message=hr_system_message,
    model=model,
    message_window_size=25,
    token_limit=8192,
)

# ==== 创建求职者 Agent ====
candidate_system_message = BaseMessage.make_assistant_message(
    role_name="Candidate",
    content=(
        "你是一名应聘Python后端工程师职位的候选人。你的背景：\n"
        "1. 有2年Python开发经验\n"
        "2. 熟悉Django和FastAPI框架\n"
        "3. 做过电商系统的后端开发\n"
        "4. 希望在新公司有更多技术成长机会\n"
        "5. 期望薪资在20-25K之间\n"
        "6. 诚实、谦虚，但也展现自己的优势\n\n"
        "请严格使用以下格式输出：\n"
        "[CANDIDATE]\n"
        "回答: <针对面试官或HR的回答>\n"
        "补充说明: <额外想说明的经验或项目>\n"
        "提问: <向面试官或HR的问题（如有）>\n"
    )
)

candidate_agent = ChatAgent(
    system_message=candidate_system_message,
    model=model,
    message_window_size=25,
    token_limit=8192,
)

# ==== 开始三人对话 ====
print("="*70)
print("💼 技术面试模拟（三人对话）")
print("="*70)
print("角色：技术面试官、HR、求职者")
print("="*70)

# HR开场
hr_msg = BaseMessage.make_assistant_message(
    role_name="HR",
    content=(
        "[HR]\n"
        "沟通内容: 您好，欢迎来到我们公司面试。今天的面试分为两部分：\n"
        "首先由技术面试官评估您的技术能力，然后我会和您聊聊职业规划。\n"
        "请先简单介绍一下自己。\n"
        "关注点: 候选人的表达能力和自信程度\n"
        "建议: 放松心态，展现真实水平"
    )
)

print(f"\n{'='*70}")
print("开场")
print(f"{'='*70}")
print(f"👔 HR:\n{hr_msg.content}\n")

# 进行多轮三人对话
conversation_history = []
last_speaker = "HR"
last_msg = hr_msg

for round_num in range(5):
    try:
        print(f"\n{'='*70}")
        print(f"第 {round_num + 1} 轮对话")
        print(f"{'='*70}")
        
        # 候选人回应（总是会说话）
        candidate_response = candidate_agent.step(last_msg)
        candidate_msg = candidate_response.msgs[0]
        print(f"\n👤 CANDIDATE:\n{candidate_msg.content}\n")
        conversation_history.append(("Candidate", candidate_msg.content))
        
        # 根据轮次决定谁来回应候选人
        if round_num % 2 == 0:
            # 技术面试官回应
            interviewer_response = interviewer_agent.step(candidate_msg)
            interviewer_msg = interviewer_response.msgs[0]
            print(f"👨‍💼 INTERVIEWER:\n{interviewer_msg.content}\n")
            conversation_history.append(("Interviewer", interviewer_msg.content))
            last_msg = interviewer_msg
            last_speaker = "Interviewer"
        else:
            # HR回应
            hr_response = hr_agent.step(candidate_msg)
            hr_msg = hr_response.msgs[0]
            print(f"👔 HR:\n{hr_msg.content}\n")
            conversation_history.append(("HR", hr_msg.content))
            last_msg = hr_msg
            last_speaker = "HR"
        
        # 检查是否结束
        if "结束" in last_msg.content or "感谢" in last_msg.content and round_num >= 3:
            print("\n✅ 面试完成")
            break
            
    except Exception as e:
        print(f"\n❌ 错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        break

print(f"\n{'='*70}")
print("面试结束")
print("="*70)

# ==== 打印对话摘要 ====
print("\n" + "="*70)
print("📊 对话摘要")
print("="*70)
print(f"总对话轮数: {len(conversation_history)}")
print(f"候选人发言次数: {len([x for x in conversation_history if x[0] == 'Candidate'])}")
print(f"面试官发言次数: {len([x for x in conversation_history if x[0] == 'Interviewer'])}")
print(f"HR发言次数: {len([x for x in conversation_history if x[0] == 'HR'])}")

