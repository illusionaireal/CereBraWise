from enum import Enum
from typing import Dict, Any, Generator, List
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableSequence
from langchain.prompts import PromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import re
import time
import os
import json

NVIDIA_API_KEY = "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"

# ---------- 状态枚举修正 ----------
class TourismState(Enum):
    INIT = "初始化"
    GET_PREFERENCE = "获取偏好"
    SELECT_SPOT = "选择景点"
    SELECT_ROUTE = "选择路线"


# ---------- 处理链构建 ----------
def build_processing_chain(prompt: PromptTemplate) -> RunnableSequence:
    """修正后的处理链"""
    llm = create_llm()
    return (
        RunnablePassthrough()
        | prompt
        | llm  # 移除了.stream直接调用
        | RunnableLambda(lambda chunk: chunk.content)  # 提取content字段
    )

# ---------- 响应生成 ----------
def format_streaming_response(content: str, options: list = None) -> Generator[Dict[str, Any], None, None]:
    """修正后的流式响应生成器"""
    # 按自然语义分割（标点结尾为分割点）
    segments = re.findall(r'.+?[。！？\n]|.+?(?=\s|$)', content)
    
    for seg in segments:
        yield {
            "messages": [{
                "role": "assistant",
                "content": seg.strip()  # 每次返回增量内容
            }],
            "options": options or []
        }
        time.sleep(0.2)  # 调整到更自然的语速

# ---------- Markdown渲染 ----------
def render_markdown(content: str) -> str:
    """安全渲染Markdown内容"""
    sanitized = re.sub(r'[^\w\s\-\.\!\?\u4e00-\u9fa5]', '', content)
    return f"```markdown\n{sanitized}\n```"

# ---------- 核心函数重构 ----------
def create_llm():
    """创建LLM实例"""
    return ChatNVIDIA(
        model="meta/llama-3.3-70b-instruct",
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
        nvidia_api_key=NVIDIA_API_KEY
    )

def state_machine(state: Dict, user_input: str) -> Generator[Dict, None, None]:
    """增强型状态机"""
    current_state = {
        "step": state.get("step", TourismState.INIT),
        "preference": state.get("preference"),
        "selected_spot": state.get("selected_spot"),
        "selected_route": state.get("selected_route"),
        "options": state.get("options", []),
        "context": state.get("context", {})  # 新增上下文存储
    }
    handlers = {
        TourismState.INIT: handle_init,
        TourismState.GET_PREFERENCE: handle_preference,
        TourismState.SELECT_SPOT: handle_spot_selection,
        TourismState.SELECT_ROUTE: handle_route_selection
    }
    
    handler = handlers.get(current_state["step"], handle_unknown)
    response_gen = handler(current_state, user_input)
    
    try:
        while True:
            chunk = next(response_gen)
            yield chunk
    except StopIteration as e:
        # 修正返回值处理
        new_state = e.value if isinstance(e.value, dict) else current_state
        yield {"messages": [{"role": "assistant", "content": ""}], "final_state": new_state}

# ---------- 状态处理函数重构 ----------
def handle_init(state: Dict, _: str) -> Generator[Dict, None, Dict]:
    """初始化处理"""
    full_response = "🏖️ 请问您想要什么样的旅游体验呢？"
    yield from format_streaming_response(full_response)
    # 明确返回更新后的状态
    return {
        **state,
        "step": TourismState.GET_PREFERENCE,
        "preference": None,
        "selected_spot": None,
        "selected_route": None
    }

# 新增解析提示模板
OPTION_PROMPTS = {
    "spot": PromptTemplate(
        input_variables=["content"],
        template="""从导游回复中提取景点名称列表：
{content}

要求：
1. 只返回JSON格式，包含spots字段的数组
2. 仅包含景点名称，不要描述
3. 示例：["景点1", "景点2", "景点3"]"""
    ),
    "route": PromptTemplate(
        input_variables=["content"],
        template="""从路线推荐中提取路线名称：
{content}

要求：
1. 只返回JSON格式，包含routes字段的数组  
2. 名称需完整包含路线特色
3. 示例：["路线1", "路线2", "路线3"]"""
    )
}

# 独立的小模型解析器
option_parser_llm = ChatNVIDIA(
    model="meta/llama3-8b-instruct",
    temperature=0.1,
    top_p=0.9,
    max_tokens=512,
    nvidia_api_key=NVIDIA_API_KEY
)

# 专用解析链
option_parser = {
    "spot": OPTION_PROMPTS["spot"] | option_parser_llm | StrOutputParser() | json.loads,
    "route": OPTION_PROMPTS["route"] | option_parser_llm | StrOutputParser() | json.loads
}

# 修改解析函数
def parse_options(content: str, option_type: str) -> List[str]:
    """使用专用提示解析选项"""
    try:
        return option_parser[option_type].invoke({"content": content})[f"{option_type}s"]
    except Exception as e:
        print(f"⚠️ 解析失败使用备用方案：{str(e)}")
        # 增强型正则匹配
        patterns = {
            "spot": r"\d+\.\s+\*\*(.*?)\*\*",
            "route": r"\d+\.\s+\*\*(.*?)\*\*|\d+\.\s+\[(.*?)\]"
        }
        return re.findall(patterns[option_type], content)

def handle_preference(state: Dict, user_input: str) -> Generator[Dict, None, Dict]:
    """动态生成景点推荐（上下文感知版）"""
    # 从知识库获取上下文（示例）
    context = {
        "spots": ["故宫", "颐和园", "天坛", "圆明园"],
        "history": "北京主要皇家景点"
    }
    
    chain = build_processing_chain(SPOT_PROMPT)
    result = chain.invoke({
        "context": context,  # 动态上下文
        "preference": user_input
    })
    
    # 流式输出处理
    yield from format_streaming_response(result)
    
    # 更新上下文
    new_context = {
        **context,
        "preference_details": user_input,
        "generated_at": time.strftime("%Y-%m-%d %H:%M")
    }
    
    return {
        **state,
        "step": TourismState.SELECT_SPOT,
        "preference": user_input,
        "options": parse_options(result, "spot"),
        "context": new_context  # 传递上下文
    }

def handle_spot_selection(state: Dict, user_input: str) -> Generator[Dict, None, Dict]:
    """动态生成路线推荐（增强版）"""
    chain = build_processing_chain(ROUTE_PROMPT)
    result = chain.invoke({"preference": state["preference"], "spot": "颐和园"})
    
    # 流式输出处理
    yield from format_streaming_response(result)
    
    # 使用小模型解析路线选项
    try:
        options = parse_options(result, "route")
    except Exception:
        options = re.findall(r"\d+\.\s+\[(.*?)\]", result)
    
    return {
        **state, 
        "step": TourismState.SELECT_ROUTE,
        "selected_spot": "颐和园",
        "options": options
    }

def handle_route_selection(state: Dict, user_input: str) -> Generator[Dict, None, Dict]:
    """动态生成攻略（增强版）"""
    chain = build_processing_chain(GUIDE_PROMPT)
    result = chain.invoke({
        "preference": state["preference"],
        "spot": state["selected_spot"],
        "route": "西堤漫步线"
    })
    
    # 流式输出处理
    yield from format_streaming_response(render_markdown(result))
    return {
        **state,
        "step": TourismState.INIT,
        "selected_route": "西堤漫步线"
    }

def handle_unknown(state: Dict, _: str) -> Generator[Dict, None, None]:
    """未知状态处理"""
    yield from format_streaming_response("⚠️ 系统遇到未知状态，正在重置...")
    return {**state, "step": TourismState.INIT}

# ---------- 完整提示模板 ----------
SPOT_PROMPT = PromptTemplate(
    input_variables=["context", "preference"],
    template="""作为资深旅行规划师，根据用户需求推荐3个景点：
    
历史背景：{context}
用户需求：{preference}

请按以下格式推荐：
1. [景点名称]：[50字特色描述]
2. [景点名称]：[50字特色描述]
3. [景点名称]：[50字特色描述]"""
)

ROUTE_PROMPT = PromptTemplate(
    input_variables=["preference", "spot"],
    template="""用户偏好：{preference}
选定景点：{spot}

请设计3种特色游览路线：
1. [路线名称]：[路线特色与亮点]
2. [路线名称]：[路线特色与亮点]
3. [路线名称]：[路线特色与亮点]"""
)

GUIDE_PROMPT = PromptTemplate(
    input_variables=["preference", "spot", "route"],
    template="""# 深度旅游攻略生成
    
用户需求：{preference}
选定景点：{spot}
选择路线：{route}

请包含以下内容：
## 行程安排（精确到小时）
## 必玩项目
## 餐饮推荐
## 预算估算
## 实用贴士"""
)

def test_llm_connection():
    """大模型连通性测试"""
    print("\n=== 开始大模型连通性测试 ===")
    try:
        llm = create_llm()
        test_prompt = PromptTemplate.from_template("请说：'测试成功'")
        chain = test_prompt | llm
        
        # 带超时的测试请求
        response = chain.invoke({"dummy": ""}).content
        if "测试成功" in response:
            print(f"✅ 大模型连接成功！响应内容：{response}")
            return True
        else:
            print(f"⚠️ 异常响应：{response}")
            return False
    except Exception as e:
        print(f"❌ 连接失败：{str(e)}")
        print("可能原因：")
        print("1. API密钥无效")
        print("2. 网络连接异常")
        print("3. 模型服务不可用")
        return False

# ---------- 新增测试验证函数 ----------
def test_initialization() -> Dict:
    """测试初始化流程"""
    print("\n=== 测试初始化 ===")
    state = {"step": TourismState.INIT}
    response_gen = state_machine(state, "")
    
    full_response = ""
    final_state = state
    try:
        for chunk in response_gen:
            if chunk.get("final_state"):
                final_state = chunk["final_state"]
                continue
            content = chunk['messages'][0]['content']
            print(f"\r系统：{full_response}{content}", end="", flush=True)
            full_response += content
        print("\n" + "-"*50)
    except StopIteration:
        pass
    
    # 验证关键字段
    assert final_state["step"] == TourismState.GET_PREFERENCE, "初始化后应进入偏好获取状态"
    print("✅ 初始化测试通过")
    return final_state

def test_preference_input(prev_state: Dict) -> Dict:
    """测试偏好输入流程"""
    print("\n=== 测试偏好输入 ===")
    user_input = "我想看皇家园林"
    response_gen = state_machine(prev_state, user_input)
    
    full_response = ""
    state = prev_state  # 初始化状态
    try:
        for chunk in response_gen:
            if chunk.get("final_state"):
                state = chunk["final_state"]
                continue
            content = chunk['messages'][0]['content']
            print(f"\r系统：{full_response}{content}", end="", flush=True)
            full_response += content
        print("\n" + "-"*50)
    except StopIteration as e:
        state = e.value if isinstance(e.value, dict) else state
    
    # 验证选项生成
    assert len(state["options"]) >= 1, "应生成至少一个景点选项"
    assert state["step"] == TourismState.SELECT_SPOT, "输入偏好后应进入景点选择状态"
    print("✅ 偏好输入测试通过")
    return state

def test_spot_selection(prev_state: Dict) -> Dict:
    """测试景点选择流程"""
    print("\n=== 测试景点选择 ===")
    user_input = "选第一个景点"
    response_gen = state_machine(prev_state, user_input)
    
    full_response = ""
    state = prev_state
    try:
        for chunk in response_gen:
            if chunk.get("final_state"):
                state = chunk["final_state"]
                continue
            content = chunk['messages'][0]['content']
            print(f"\r系统：{full_response}{content}", end="", flush=True)
            full_response += content
        print("\n" + "-"*50)
    except StopIteration as e:
        state = e.value if isinstance(e.value, dict) else state
    
    assert "selected_spot" in state, "应记录已选景点"
    assert len(state["options"]) >= 1, "应生成至少一个路线选项"
    print("✅ 景点选择测试通过")
    return state

def test_route_selection(prev_state: Dict) -> Dict:
    """测试路线选择流程"""
    print("\n=== 测试路线选择 ===")
    user_input = "选文化路线"
    response_gen = state_machine(prev_state, user_input)
    
    full_response = ""
    state = prev_state  # 关键修复：初始化状态
    try:
        for chunk in response_gen:
            if chunk.get("final_state"):
                state = chunk["final_state"]
                continue
            content = chunk['messages'][0]['content']
            print(f"\r系统：{full_response}{content}", end="", flush=True)
            full_response += content
        print("\n" + "-"*50)
    except StopIteration as e:
        state = e.value if isinstance(e.value, dict) else state
    
    # 验证最终状态
    assert state["step"] == TourismState.INIT, "完成流程后应重置状态"
    assert "selected_route" in state, "应记录已选路线"
    print("✅ 路线选择测试通过")
    return state

# 新增交互演示函数
def interactive_demo():
    """命令行交互演示"""
    print("🛫 旅游助手 交互模式（输入exit退出）")
    state = {"step": TourismState.INIT}
    
    while True:
        try:
            # 根据状态提示输入
            if state["step"] == TourismState.INIT:
                user_input = input("\n> 按回车开始规划旅程：") or ""
            elif state["step"] == TourismState.GET_PREFERENCE:
                user_input = input("\n> 请描述您的旅游偏好（如：想看皇家建筑）：")
            elif state["step"] == TourismState.SELECT_SPOT:
                print("\n推荐景点：")
                for i, spot in enumerate(state["options"], 1):
                    print(f"  {i}. {spot}")
                user_input = input("> 请选择景点编号或名称：")
            elif state["step"] == TourismState.SELECT_ROUTE:
                print("\n推荐路线：")
                for i, route in enumerate(state["options"], 1):
                    print(f"  {i}. {route}")
                user_input = input("> 请选择路线编号或名称：")
            
            if user_input.lower() == "exit":
                break

            # 处理输入
            full_response = ""
            response_gen = state_machine(state, user_input)
            for chunk in response_gen:
                if chunk.get("final_state"):
                    state = chunk["final_state"]
                    continue
                content = chunk['messages'][0]['content']
                print(f"\r系统：{full_response}{content}", end="", flush=True)
                full_response += content
            
            # 显示最终结果
            if state["step"] == TourismState.INIT and "selected_route" in state:
                print("\n\n✅ 行程规划完成！")
                print("-"*50)
                print(full_response)
                print("-"*50)
                state = {"step": TourismState.INIT}  # 重置状态

        except KeyboardInterrupt:
            print("\n\n👋 感谢使用，再见！")
            break

# ---------- 更新主流程 ----------
if __name__ == "__main__":
    # 初始化大模型
    llm = create_llm()
    print("✅ 大模型初始化成功")
    
    # 启动交互模式
    interactive_demo()

