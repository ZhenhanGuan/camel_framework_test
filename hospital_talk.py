# demo_openai_compatible_single.py
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

# 🔧 修复：确保 BASE_URL 以 /v1 结尾（OpenAI 兼容接口需要）
if BASE_URL and not BASE_URL.endswith('/v1'):
    BASE_URL = BASE_URL.rstrip('/') + '/v1'

model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type=MODEL_ID,
    api_key=API_KEY,
    url=BASE_URL,
    model_config_dict={
        "temperature": 0.9,
        "max_tokens": 1500,  # 增加每次回复的最大 token 数
    },
)

# ==== 创建医生 Agent ====
doctor_system_message = BaseMessage.make_assistant_message(
    role_name="Doctor",
    content=(
        "你是一名专业的医生。你的职责是：\n"
        "1. 仔细询问患者的症状、病史和生活习惯\n"
        "2. 根据患者描述进行初步诊断\n"
        "3. 给出专业的医疗建议和治疗方案\n"
        "4. 用通俗易懂的语言解释医学概念\n"
        "5. 保持耐心、专业和同理心\n\n"
        "请严格使用以下格式输出：\n"
        "[DOCTOR]\n"
        "本轮目标: <说明本轮沟通目标>\n"
        "询问/说明: <向患者询问的问题或医学解释>\n"
        "初步判断: <基于已知信息的分析>\n"
        "建议: <检查项目或治疗方案>\n"
        "注意事项: <患者需要注意的要点>\n"
    )
)

doctor_agent = ChatAgent(
    system_message=doctor_system_message,
    model=model,
    message_window_size=20,
    token_limit=8192,  # 增加 token 限制
)

# ==== 创建患者 Agent ====
patient_system_message = BaseMessage.make_user_message(
    role_name="Patient",
    content=(
        "你是一名因头痛来就诊的患者。你的特点是：\n"
        "1. 头痛已经持续3天，主要在太阳穴位置\n"
        "2. 最近工作压力大，经常熬夜\n"
        "3. 对自己的病情有些担心\n"
        "4. 会如实回答医生的问题\n"
        "5. 对不理解的医学术语会提问\n\n"
        "请严格使用以下格式输出：\n"
        "[PATIENT]\n"
        "症状描述: <详细描述不适症状>\n"
        "回答医生: <针对医生问题的具体回答>\n"
        "疑问/顾虑: <对病情或治疗的疑问>\n"
    )
)

patient_agent = ChatAgent(
    system_message=patient_system_message,
    model=model,
    message_window_size=20,
    token_limit=8192,  # 增加 token 限制
)

# ==== 开始对话 ====
print("="*70)
print("🏥 医患沟通模拟（独立 Agent 版本）")
print("="*70)

# 患者主动开始对话
patient_msg = BaseMessage.make_user_message(
    role_name="Patient",
    content="医生您好，我最近头痛得厉害，已经持续3天了。"
)

print(f"\n{'='*70}")
print("初始消息")
print(f"{'='*70}")
print(f"🤒 PATIENT: {patient_msg.content}\n")

# 进行多轮对话
for i in range(6):
    try:
        print(f"\n{'='*70}")
        print(f"第 {i+1} 轮对话")
        print(f"{'='*70}")
        
        # 医生回应患者
        doctor_response = doctor_agent.step(patient_msg)
        doctor_msg = doctor_response.msgs[0]
        print(f"\n👨‍⚕️ DOCTOR:\n{doctor_msg.content}\n")
        
        # 患者回应医生
        patient_response = patient_agent.step(doctor_msg)
        patient_msg = patient_response.msgs[0]
        print(f"🤒 PATIENT:\n{patient_msg.content}\n")
        
        # 检查是否结束
        if "再见" in doctor_msg.content or "结束" in doctor_msg.content:
            print("\n✅ 问诊完成")
            break
            
    except Exception as e:
        print(f"\n❌ 错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        break

print(f"\n{'='*70}")
print("问诊结束")
print("="*70)

