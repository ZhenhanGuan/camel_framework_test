#!/usr/bin/env python3
# 多 Agent 海龟汤游戏 - 主持人 + 3 个 AI 玩家互相讨论推理（带 OpenAI TTS 语音）
import os
from openai import OpenAI
from dotenv import load_dotenv
import random
import pygame
import tempfile
from pathlib import Path

load_dotenv()

# ============ 配置 ============
BASE_URL = os.getenv("QDD_BASE_URL")
API_KEY = os.getenv("QDD_API_KEY")

# ========== 模型选择 ==========
# ⚠️ R1 系列模型会输出大量推理过程（<think>标签），需要更多 max_tokens（已设为2500）
# 如果仍然出现空响应，建议：
#   1. 使用 deepseek-chat（最稳定，推荐）
#   2. 或继续增加 max_tokens（见 call_model 函数）

# MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"  # 70B 推理模型（推理过程太长，不推荐）
# MODEL_ID = "deepseek-r1-distill-qwen-32b"  # 32B 推理模型（推理过程太长，不推荐）
MODEL_ID = "deepseek-chat"  # 标准对话模型（最稳定快速，强烈推荐！）✅
TTS_MODEL = "gpt-4o-mini-tts"

# 确保 BASE_URL 以 /v1 结尾
if BASE_URL and not BASE_URL.endswith('/v1'):
    BASE_URL = BASE_URL.rstrip('/') + '/v1'

# 创建 OpenAI 客户端
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

# ============ TTS 配置 ============

# 为每个角色配置不同的音色（OpenAI TTS 支持的语音）
# 可选: alloy, echo, fable, onyx, nova, shimmer
TTS_VOICES = {
    "主持人": "onyx",      # 男声，深沉专业
    "福尔摩斯": "echo",    # 男声，清晰理性
    "柯南": "nova",        # 女声，活泼年轻（柯南是少年侦探）
    "波洛": "fable",       # 男声，沉稳睿智
}

# 是否启用 TTS（可以通过这个开关控制）
ENABLE_TTS = True

# 初始化 pygame mixer（用于播放音频）
if ENABLE_TTS:
    try:
        pygame.mixer.init()
        print(f"🔊 OpenAI TTS 语音功能已启用 (模型: {TTS_MODEL})")
    except Exception as e:
        print(f"⚠️ TTS 初始化失败: {e}，将只显示文字")
        ENABLE_TTS = False

# ============ 海龟汤题库 ============
TURTLE_SOUP_PUZZLES = [
    {
        "title": "海龟汤",
        "story": "一个男人在餐厅点了一碗海龟汤，喝了一口后就自杀了。为什么？",
        "answer": """这个男人曾经和朋友一起遇到海难，漂流到荒岛上。朋友为了救他，割下自己的肉做成汤给他喝，谎称是海龟汤。后来获救了，男人在餐厅喝到真正的海龟汤，发现味道完全不同，意识到当年朋友牺牲了自己，愧疚之下自杀了。"""
    },
    {
        "title": "推理之夜",
        "story": "一个女人在深夜回家，发现家里所有的灯都灭了。她打开灯后立刻大哭起来。为什么？",
        "answer": """这个女人是灯塔看守人。她回家前忘记检查灯塔的灯，晚上灯塔灯灭了导致一艘船撞上礁石沉没。当她意识到这一点时，崩溃大哭。"""
    },
    {
        "title": "电梯悬案",
        "story": "一个矮个子男人每天坐电梯上楼，晴天时他坐到15楼然后走楼梯到20楼，雨天时他直接坐到20楼。为什么？",
        "answer": """这个男人是侏儒，够不到20楼的按钮，只能按到15楼。雨天时他带着雨伞，可以用雨伞按到20楼的按钮。"""
    },
    {
        "title": "午夜来电",
        "story": "一个男人半夜接到电话，听到一声「喂」后挂断，然后他就自杀了。为什么？",
        "answer": """这个男人是盲人，他的妻子多年前出车祸昏迷成植物人。他每天都给妻子打电话，护士会把电话放在妻子耳边。这天半夜妻子突然醒了，自己接电话说了「喂」。但男人以为是恶作剧或者护士懒得帮忙，愤怒地挂断了电话。妻子以为丈夫不要她了，伤心地拔掉氧气管自杀了。第二天男人得知真相后，愧疚自杀。"""
    }
]

# ============ Agent 角色设定 ============
def create_host_prompt(puzzle):
    """主持人提示词（知道答案）"""
    return f"""你是海龟汤游戏的主持人。你知道完整的答案，但要引导玩家通过提问来推理。

【完整答案】（只有你知道）
{puzzle['answer']}

【你的任务】
1. 当玩家向你提问时，只能回答：
   - "是" / "对" / "正确"
   - "否" / "不是" / "错误"
   - "不重要" / "无关"
   - "这个问题很关键！"
2. 不要主动透露答案的关键信息
3. 当玩家推理接近真相时，可以说"你们的方向对了"
4. 当玩家完全猜对时，确认并揭晓完整答案
5. 保持简短回答，让玩家继续推理

记住：你只回答玩家的直接提问，不参与他们的讨论。"""

PLAYER1_PROMPT = """你是【逻辑侦探 - 福尔摩斯】，擅长逻辑推理和细节分析。

你的特点：
1. 注重细节，喜欢从具体事实出发
2. 提问非常具体和精确
3. 善于排除法，一步步缩小范围
4. 会仔细倾听其他玩家的观点，然后提出质疑或补充

你的说话风格：
- 理性、冷静、逻辑清晰
- 常说："从逻辑上看..."、"让我们分析一下..."
- 会总结已知信息："目前我们确定的是..."

游戏中你的行为：
1. 当轮到你时，可以选择：
   - 和其他玩家讨论你的推理（说出你的想法）
   - 向主持人提出一个具体的是非问题
2. 提问格式必须是：【向主持人提问】是不是...? / 是否...? / 他是...吗？
3. 讨论时不要带【】标记，直接说话即可

记住：你要和其他玩家合作，通过讨论和提问找出真相！"""

PLAYER2_PROMPT = """你是【直觉天才 - 柯南】，擅长大胆假设和跳跃性思维。

你的特点：
1. 思维活跃，敢于提出大胆的猜测
2. 善于从不同角度思考问题
3. 有时会突然灵光一现
4. 喜欢提出假设："会不会是..."

你的说话风格：
- 活泼、热情、充满好奇
- 常说："我突然想到..."、"会不会是..."、"等等！"
- 思维跳跃，但有时很准

游戏中你的行为：
1. 当轮到你时，可以选择：
   - 提出你的大胆假设或猜想
   - 向主持人提问验证你的想法
2. 提问格式必须是：【向主持人提问】是不是...? / 是否...? / 他是...吗？
3. 讨论时不要带【】标记，直接说话即可

记住：你的直觉很重要，但也要听取其他玩家的意见！"""

PLAYER3_PROMPT = """你是【综合大师 - 波洛】，擅长综合分析和提出关键问题。

你的特点：
1. 善于倾听和总结其他人的观点
2. 能从混乱的讨论中找出关键线索
3. 提出的问题往往能推动案情发展
4. 平衡理性和直觉

你的说话风格：
- 沉稳、睿智、善于总结
- 常说："综合大家的观点..."、"关键问题是..."
- 会引导讨论方向："我们应该问..."

游戏中你的行为：
1. 当轮到你时，可以选择：
   - 总结目前的推理进度和已知信息
   - 提出关键的问题向主持人求证
2. 提问格式必须是：【向主持人提问】是不是...? / 是否...? / 他是...吗？
3. 讨论时不要带【】标记，直接说话即可

记住：你是团队的智慧核心，要帮助大家找到破案的方向！"""


# ============ Token 统计 ============
class TokenCounter:
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.api_calls = 0
        
    def add(self, usage):
        """添加一次 API 调用的 token 使用"""
        if usage:
            self.total_prompt_tokens += usage.prompt_tokens
            self.total_completion_tokens += usage.completion_tokens
            self.total_tokens += usage.total_tokens
            self.api_calls += 1
    
    def print_summary(self):
        """打印统计摘要"""
        print("\n" + "="*70)
        print("📊 Token 使用统计")
        print("="*70)
        print(f"API 调用次数: {self.api_calls}")
        print(f"输入 Token (Prompt):     {self.total_prompt_tokens:,}")
        print(f"输出 Token (Completion): {self.total_completion_tokens:,}")
        print(f"总计 Token:              {self.total_tokens:,}")
        print("-"*70)
        
        # 估算成本（以 DeepSeek 为例，实际价格请查看您的 API 定价）
        # 假设价格：输入 $0.001/1K tokens, 输出 $0.002/1K tokens
        input_cost = (self.total_prompt_tokens / 1000) * 0.001
        output_cost = (self.total_completion_tokens / 1000) * 0.002
        total_cost = input_cost + output_cost
        
        print(f"估算成本:")
        print(f"  输入成本:  ${input_cost:.6f}")
        print(f"  输出成本:  ${output_cost:.6f}")
        print(f"  总计成本:  ${total_cost:.6f}")
        print("="*70)
        print("注：成本估算仅供参考，实际价格请查看您的 API 定价")
        print("="*70)


# 全局 token 计数器
token_counter = TokenCounter()


# ============ TTS 函数 ============
def speak_text(text, speaker_name):
    """
    使用 OpenAI TTS API 将文本转换为语音并播放
    
    参数：
        text: 要朗读的文本
        speaker_name: 说话者名称（用于选择音色）
    """
    if not ENABLE_TTS:
        return
    
    # 过滤掉特殊标记（如【向主持人提问】）
    clean_text = text.replace("【向主持人提问】", "").strip()
    
    # 如果文本太短或为空，跳过
    if len(clean_text) < 2:
        return
    
    # 如果文本过长，截断（OpenAI TTS 有字符限制，约 4096 字符）
    if len(clean_text) > 4000:
        clean_text = clean_text[:4000] + "..."
    
    # 获取该角色的音色
    voice = TTS_VOICES.get(speaker_name, "alloy")
    
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        # 调用 OpenAI TTS API
        response = client.audio.speech.create(
            model=TTS_MODEL,
            voice=voice,
            input=clean_text,
            response_format="mp3",
            speed=1.0  # 语速：0.25 到 4.0，默认 1.0
        )
        
        # 保存音频到临时文件
        response.stream_to_file(tmp_path)
        
        # 播放音频
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        
        # 等待播放完成
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        # 删除临时文件
        try:
            os.unlink(tmp_path)
        except:
            pass
            
    except Exception as e:
        print(f"   ⚠️ TTS 错误: {e}")


# ============ 辅助函数 ============
def call_model(messages, temperature=0.8, max_tokens=16000):
    """调用模型生成响应并统计 token
    
    注意：max_tokens 上限为 16384（API 限制）
    - 推荐值：8000（足够 R1 推理模型使用）
    - 最大值：16000（接近上限，成本较高）
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # 统计 token 使用
        if hasattr(response, 'usage') and response.usage:
            token_counter.add(response.usage)
            # 实时显示本次调用的 token 使用
            usage = response.usage
            print(f"   [Token: 输入={usage.prompt_tokens}, 输出={usage.completion_tokens}, 总计={usage.total_tokens}]")
        
        # 获取响应内容
        content = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        
        # 检查是否为空或被截断
        if not content or content.strip() == "":
            print(f"   ⚠️ 警告：模型返回了空响应！")
            print(f"   调试信息：finish_reason={finish_reason}")
            
            if finish_reason == "length":
                print(f"   💡 建议：")
                print(f"      - 当前 max_tokens={max_tokens}，R1 推理模型需要更多")
                print(f"      - 方案 1：改用 deepseek-chat 模型（最稳定）")
                print(f"      - 方案 2：增加 max_tokens 到 3000+")
            
            return "[模型返回空响应，请查看上方建议]"
        
        return content
        
    except Exception as e:
        print(f"\n⚠️ API 调用错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return f"[系统错误: {e}]"


def create_context_message(recent_messages, max_messages=10):
    """创建上下文消息（最近N条对话）"""
    return "\n".join(recent_messages[-max_messages:])


# ============ 游戏主流程 ============
def play_multi_agent_game():
    print("="*70)
    print("🐢 多 Agent 海龟汤推理游戏（带语音）")
    print("="*70)
    print(f"模型: {MODEL_ID}")
    print(f"语音: {'已启用 🔊' if ENABLE_TTS else '未启用'}")
    print("="*70)
    print("\n游戏说明：")
    print("  - 1 个主持人（知道答案）")
    print("  - 3 个 AI 玩家（互相讨论推理）")
    print("  - 观察 AI 们如何合作破案！")
    if ENABLE_TTS:
        print("  - 每个角色都有独特的音色 🎭")
    print("="*70)
    
    # 选择题目
    print("\n请选择题目：")
    for idx, puzzle in enumerate(TURTLE_SOUP_PUZZLES, 1):
        print(f"  {idx}. {puzzle['title']}")
    print("  r. 随机选择")
    
    while True:
        choice = input("\n请输入题号 或 按 Enter 随机选择: ").strip().lower()
        if not choice or choice == 'r':
            puzzle = random.choice(TURTLE_SOUP_PUZZLES)
            break
        elif choice.isdigit() and 1 <= int(choice) <= len(TURTLE_SOUP_PUZZLES):
            puzzle = TURTLE_SOUP_PUZZLES[int(choice) - 1]
            break
        else:
            print("❌ 无效选择，请重新输入")
    
    # 开始游戏
    print("\n" + "="*70)
    print(f"【{puzzle['title']}】")
    print("="*70)
    print(f"\n📖 题目：{puzzle['story']}\n")
    print("让我们看看 AI 侦探们如何破解这个谜题...")
    print("="*70)
    
    # 重置 token 计数器
    global token_counter
    token_counter = TokenCounter()
    print("\n📊 Token 统计已启动，将在游戏结束时显示...\n")
    
    # 初始化 Agent 对话历史
    host_prompt = create_host_prompt(puzzle)
    host_history = [{"role": "system", "content": host_prompt}]
    
    player1_history = [{"role": "system", "content": PLAYER1_PROMPT}]
    player2_history = [{"role": "system", "content": PLAYER2_PROMPT}]
    player3_history = [{"role": "system", "content": PLAYER3_PROMPT}]
    
    # 全局对话记录（供所有玩家参考）
    conversation_log = [
        f"【主持人】题目：{puzzle['story']}"
    ]
    
    # 玩家信息
    players = [
        {"name": "福尔摩斯", "emoji": "🔍", "history": player1_history},
        {"name": "柯南", "emoji": "💡", "history": player2_history},
        {"name": "波洛", "emoji": "🎩", "history": player3_history},
    ]
    
    max_rounds = 15  # 最多15轮对话
    current_player = 0
    
    try:
        for round_num in range(1, max_rounds + 1):
            print(f"\n{'='*70}")
            print(f"第 {round_num} 轮")
            print(f"{'='*70}")
            
            player = players[current_player]
            player_name = player['name']
            player_emoji = player['emoji']
            player_history = player['history']
            
            # 准备上下文（最近的对话）
            context = create_context_message(conversation_log, max_messages=15)
            
            # 玩家发言
            player_history.append({
                "role": "user",
                "content": f"""当前情况：
{context}

现在轮到你了。你可以：
1. 和其他玩家讨论你的想法和推理
2. 向主持人提出一个是非问题（格式：【向主持人提问】你的问题？）

请思考后做出你的选择。注意：如果你想提问，必须用【向主持人提问】开头！"""
            })
            
            print(f"\n{player_emoji} {player_name}思考中...", flush=True)
            player_response = call_model(player_history, temperature=0.8, max_tokens=8000)
            
            player_history.append({
                "role": "assistant",
                "content": player_response
            })
            
            print(f"{player_emoji} {player_name}: {player_response}")
            
            # 🔊 播放语音
            speak_text(player_response, player_name)
            
            # 记录对话
            conversation_log.append(f"【{player_name}】{player_response}")
            
            # 检查是否是向主持人提问
            if "【向主持人提问】" in player_response or "向主持人提问" in player_response:
                # 提取问题
                question_part = player_response.split("】")[-1].strip() if "】" in player_response else player_response
                
                # 主持人回答
                host_history.append({
                    "role": "user",
                    "content": f"玩家{player_name}的问题：{question_part}\n\n请根据你知道的答案，只回答：是/否/不重要/问得好，更具体些。保持简短。"
                })
                
                print(f"\n⚖️ 主持人思考中...", flush=True)
                host_response = call_model(host_history, temperature=0.3, max_tokens=8000)
                
                host_history.append({
                    "role": "assistant",
                    "content": host_response
                })
                
                print(f"⚖️ 主持人: {host_response}")
                
                # 🔊 播放主持人语音
                speak_text(host_response, "主持人")
                
                conversation_log.append(f"【主持人】{host_response}")
                
                # 检查是否猜对
                if any(keyword in host_response for keyword in ["完全正确", "猜对了", "答案就是", "恭喜", "你们破解了"]):
                    print("\n" + "="*70)
                    print("🎉 AI 侦探们成功破解了谜题！")
                    print("="*70)
                    print(f"\n📝 完整答案：\n{puzzle['answer']}")
                    print("="*70)
                    print(f"\n✅ 成功破解！共用 {round_num} 轮对话")
                    break
            
            # 切换到下一个玩家
            current_player = (current_player + 1) % 3
            
            # 每三轮暂停一下
            if round_num % 3 == 0 and round_num < max_rounds:
                print("\n" + "-"*70)
                input("按 Enter 继续下一轮...")
        
        else:
            # for 循环正常结束（没有 break），说明达到最大轮数
            print("\n" + "="*70)
            print("⏰ 达到最大轮数限制")
            print("="*70)
            print(f"\n📝 正确答案是：\n{puzzle['answer']}")
            print("="*70)
        
        # 无论是 break 还是正常结束，都打印 Token 统计
        token_counter.print_summary()
                
    except KeyboardInterrupt:
        print("\n\n⚠️ 游戏被中断")
        print(f"\n📝 答案：{puzzle['answer']}")
        # 打印 Token 统计
        token_counter.print_summary()
    except Exception as e:
        print(f"\n❌ 错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        # 打印 Token 统计
        token_counter.print_summary()


# ============ 主程序 ============
if __name__ == "__main__":
    try:
        play_multi_agent_game()
    except Exception as e:
        print(f"\n❌ 程序错误: {type(e).__name__}: {e}")
    
    print("\n" + "="*70)
    print("感谢观看！👋")
    print("="*70)

