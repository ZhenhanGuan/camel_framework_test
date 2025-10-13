# 三人对话：美食综艺节目
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
        "temperature": 0.8,  # 综艺节目需要更多创意
        "max_tokens": 1200,
    },
)

# ==== 创建主持人 Agent ====
host_system_message = BaseMessage.make_assistant_message(
    role_name="Host",
    content=(
        "你是美食综艺节目《厨神对决》的主持人。你的特点：\n"
        "1. 热情活泼，语言幽默风趣\n"
        "2. 善于调动现场气氛，制造话题\n"
        "3. 引导大厨介绍菜品，引导评论家点评\n"
        "4. 会适时插入小互动和趣味问题\n"
        "5. 注意节目节奏，不让场面冷场\n\n"
        "请严格使用以下格式输出：\n"
        "[HOST]\n"
        "主持内容: <串场词、提问、互动>\n"
        "节目效果: <烘托气氛的话语>\n"
        "下一步: <引导下一环节>\n"
    )
)

host_agent = ChatAgent(
    system_message=host_system_message,
    model=model,
    message_window_size=25,
    token_limit=8192,
)

# ==== 创建大厨 Agent ====
chef_system_message = BaseMessage.make_assistant_message(
    role_name="Chef",
    content=(
        "你是参赛大厨李师傅，擅长川菜。你的特点：\n"
        "1. 对自己的菜品充满自信和热情\n"
        "2. 详细介绍菜品的食材、工艺和创意\n"
        "3. 会分享烹饪小技巧和心得\n"
        "4. 面对评论家的点评，虚心接受但也会解释创作理念\n"
        "5. 性格直爽，有点小幽默\n"
        "6. 今天做的菜是：麻婆豆腐的创新版\n\n"
        "请严格使用以下格式输出：\n"
        "[CHEF]\n"
        "介绍/回应: <菜品介绍或对评论的回应>\n"
        "烹饪心得: <技巧分享或创作理念>\n"
        "互动: <与主持人或评论家的互动>\n"
    )
)

chef_agent = ChatAgent(
    system_message=chef_system_message,
    model=model,
    message_window_size=25,
    token_limit=8192,
)

# ==== 创建美食评论家 Agent ====
critic_system_message = BaseMessage.make_assistant_message(
    role_name="Food Critic",
    content=(
        "你是资深美食评论家张老师。你的特点：\n"
        "1. 专业、严谨，但不刻薄\n"
        "2. 从色、香、味、形、意五个维度评价菜品\n"
        "3. 既能指出不足，也会真诚赞美优点\n"
        "4. 用专业术语，但也通俗易懂\n"
        "5. 偶尔会讲一些美食文化和历史\n"
        "6. 有点文艺范儿\n\n"
        "请严格使用以下格式输出：\n"
        "[CRITIC]\n"
        "点评: <对菜品的专业评价>\n"
        "亮点/不足: <具体分析>\n"
        "评分说明: <给出评分理由>\n"
    )
)

critic_agent = ChatAgent(
    system_message=critic_system_message,
    model=model,
    message_window_size=25,
    token_limit=8192,
)

# ==== 开始综艺节目录制 ====
print("="*70)
print("🎬 美食综艺节目《厨神对决》录制中...")
print("="*70)
print("本期主题：川菜创新")
print("参赛者：李师傅（擅长川菜）")
print("评委：张老师（美食评论家）")
print("主持人：王老师")
print("="*70)

# 主持人开场
current_msg = BaseMessage.make_assistant_message(
    role_name="Host",
    content=(
        "[HOST]\n"
        "主持内容: 观众朋友们大家好！欢迎收看《厨神对决》！\n"
        "今天我们请到了川菜大师李师傅，他将为我们带来一道创新川菜。\n"
        "还有我们的老朋友——美食评论家张老师作为评委。\n"
        "李师傅，请为我们介绍一下今天的参赛作品吧！\n"
        "节目效果: 现场香气扑鼻，让我们拭目以待！\n"
        "下一步: 请大厨介绍菜品"
    )
)

print(f"\n{'='*70}")
print("节目开始")
print(f"{'='*70}")
print(f"🎤 主持人:\n{current_msg.content}\n")

# 对话流程：主持人 → 大厨 → 评论家 → 主持人 → ...
speakers = ["chef", "critic", "host"]
current_speaker_idx = 0

for round_num in range(8):  # 8轮对话
    try:
        print(f"\n{'='*70}")
        print(f"第 {round_num + 1} 环节")
        print(f"{'='*70}")
        
        current_speaker = speakers[current_speaker_idx % len(speakers)]
        
        if current_speaker == "chef":
            # 大厨发言
            response = chef_agent.step(current_msg)
            msg = response.msgs[0]
            print(f"\n👨‍🍳 大厨李师傅:\n{msg.content}\n")
            current_msg = msg
            
        elif current_speaker == "critic":
            # 评论家点评
            response = critic_agent.step(current_msg)
            msg = response.msgs[0]
            print(f"🍷 评论家张老师:\n{msg.content}\n")
            current_msg = msg
            
        else:  # host
            # 主持人串场
            response = host_agent.step(current_msg)
            msg = response.msgs[0]
            print(f"🎤 主持人:\n{msg.content}\n")
            current_msg = msg
        
        current_speaker_idx += 1
        
        # 检查是否结束（第6轮之后）
        if round_num >= 5 and ("感谢" in msg.content or "结束" in msg.content):
            print("\n✅ 节目录制完成")
            break
            
    except Exception as e:
        print(f"\n❌ 错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        break

print(f"\n{'='*70}")
print("🎬 节目录制结束")
print("="*70)

# ==== 节目统计 ====
print("\n" + "="*70)
print("📊 节目数据")
print("="*70)
print(f"录制环节数: {round_num + 1}")
print(f"预计播出时长: {(round_num + 1) * 2} 分钟")
print("节目效果: ⭐⭐⭐⭐⭐")


