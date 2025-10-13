# 三人对话：辩论赛
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
        "temperature": 0.8,  # 辩论需要较高创意
        "max_tokens": 1500,
    },
)

# ==== 创建主持人 Agent ====
moderator_system_message = BaseMessage.make_assistant_message(
    role_name="Moderator",
    content=(
        "你是辩论赛的主持人。你的职责：\n"
        "1. 保持中立，不偏袒任何一方\n"
        "2. 控制辩论节奏和时间\n"
        "3. 引导双方围绕核心问题展开辩论\n"
        "4. 适时总结双方观点\n"
        "5. 提出关键问题让双方深入讨论\n"
        "6. 维持辩论秩序和礼仪\n\n"
        "请严格使用以下格式输出：\n"
        "[MODERATOR]\n"
        "主持内容: <串场、提问、规则说明>\n"
        "观点总结: <总结双方已提出的观点>\n"
        "下一环节: <引导下一步>\n"
    )
)

moderator_agent = ChatAgent(
    system_message=moderator_system_message,
    model=model,
    message_window_size=30,
    token_limit=10240,
)

# ==== 创建正方辩手 Agent ====
pro_system_message = BaseMessage.make_assistant_message(
    role_name="Pro Side",
    content=(
        "你是辩论赛正方辩手，立场：【人工智能的发展利大于弊】\n\n"
        "你的特点：\n"
        "1. 论点清晰，逻辑严密\n"
        "2. 用数据、案例、理论支持观点\n"
        "3. 积极驳斥反方论点，找出其逻辑漏洞\n"
        "4. 强调AI在医疗、教育、科研等领域的贡献\n"
        "5. 论述AI提高效率、解放人类创造力\n"
        "6. 保持礼貌但态度坚定\n\n"
        "请严格使用以下格式输出：\n"
        "[PRO]\n"
        "立论/驳论: <陈述观点或反驳对方>\n"
        "论据支撑: <数据、案例、理论>\n"
        "小结: <强化本方立场>\n"
    )
)

pro_agent = ChatAgent(
    system_message=pro_system_message,
    model=model,
    message_window_size=30,
    token_limit=10240,
)

# ==== 创建反方辩手 Agent ====
con_system_message = BaseMessage.make_assistant_message(
    role_name="Con Side",
    content=(
        "你是辩论赛反方辩手，立场：【人工智能的发展弊大于利】\n\n"
        "你的特点：\n"
        "1. 论点犀利，能抓住关键问题\n"
        "2. 用反例、风险、道德困境质疑AI\n"
        "3. 反驳正方论据，指出其片面性\n"
        "4. 强调AI带来的失业、隐私、伦理风险\n"
        "5. 论述人类对AI失控的担忧\n"
        "6. 保持理性但立场鲜明\n\n"
        "请严格使用以下格式输出：\n"
        "[CON]\n"
        "立论/驳论: <陈述观点或反驳对方>\n"
        "论据支撑: <反例、风险分析、逻辑推理>\n"
        "小结: <强化本方立场>\n"
    )
)

con_agent = ChatAgent(
    system_message=con_system_message,
    model=model,
    message_window_size=30,
    token_limit=10240,
)

# ==== 开始辩论赛 ====
print("="*70)
print("🎓 辩论赛：人工智能的发展是利大于弊还是弊大于利？")
print("="*70)
print("正方观点：人工智能的发展利大于弊")
print("反方观点：人工智能的发展弊大于利")
print("主持人：保持中立，引导辩论")
print("="*70)

# 主持人开场
current_msg = BaseMessage.make_assistant_message(
    role_name="Moderator",
    content=(
        "[MODERATOR]\n"
        "主持内容: 各位观众，欢迎来到本场辩论赛！\n"
        "今天的辩题是：人工智能的发展是利大于弊还是弊大于利？\n"
        "正方认为利大于弊，反方认为弊大于利。\n"
        "辩论分为：开篇立论、攻辩、自由辩论、总结陈词四个环节。\n"
        "首先请正方进行开篇立论，时间3分钟。\n"
        "观点总结: 辩论尚未开始\n"
        "下一环节: 正方开篇立论"
    )
)

print(f"\n{'='*70}")
print("【开场】")
print(f"{'='*70}")
print(f"⚖️ 主持人:\n{current_msg.content}\n")

# 辩论流程设计
debate_stages = [
    ("正方立论", "pro"),
    ("反方立论", "con"),
    ("主持人提问", "moderator"),
    ("正方回应", "pro"),
    ("反方反驳", "con"),
    ("主持人引导", "moderator"),
    ("正方深入论述", "pro"),
    ("反方深入论述", "con"),
    ("主持人总结", "moderator"),
]

for stage_num, (stage_name, speaker) in enumerate(debate_stages):
    try:
        print(f"\n{'='*70}")
        print(f"【{stage_name}】 - 第 {stage_num + 1} 环节")
        print(f"{'='*70}")
        
        if speaker == "pro":
            # 正方发言
            response = pro_agent.step(current_msg)
            msg = response.msgs[0]
            print(f"\n✅ 正方辩手:\n{msg.content}\n")
            current_msg = msg
            
        elif speaker == "con":
            # 反方发言
            response = con_agent.step(current_msg)
            msg = response.msgs[0]
            print(f"❌ 反方辩手:\n{msg.content}\n")
            current_msg = msg
            
        else:  # moderator
            # 主持人发言
            response = moderator_agent.step(current_msg)
            msg = response.msgs[0]
            print(f"⚖️ 主持人:\n{msg.content}\n")
            current_msg = msg
            
    except Exception as e:
        print(f"\n❌ 错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        break

print(f"\n{'='*70}")
print("🎓 辩论赛结束")
print("="*70)

# ==== 辩论统计 ====
print("\n" + "="*70)
print("📊 辩论数据")
print("="*70)
print(f"辩论环节数: {len(debate_stages)}")
print(f"正方发言次数: {len([x for x in debate_stages if x[1] == 'pro'])}")
print(f"反方发言次数: {len([x for x in debate_stages if x[1] == 'con'])}")
print(f"主持人发言次数: {len([x for x in debate_stages if x[1] == 'moderator'])}")
print("\n辩论核心议题：")
print("1. AI对就业的影响")
print("2. AI的伦理与安全问题")
print("3. AI对人类社会的整体价值")


