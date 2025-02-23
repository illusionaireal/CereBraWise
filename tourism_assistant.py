#coding=UTF-8
from enum import Enum
import langchain
from typing import Dict, Any, Generator, List
from langchain.output_parsers import RetryOutputParser  # 正确导入路径
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableSequence, RunnableBranch, \
    RunnableAssign, RunnableParallel
from langchain.prompts import PromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import re
import time
import os
import json
import img_preprocess
# 新增Pydantic模型
from pydantic import BaseModel, Field
from typing import List
# 修改解析器配置
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser

NVIDIA_API_KEY = "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"


# ---------- 状态枚举修正 ----------
class TourismState(Enum):
    INIT = "初始化"
    SELECT_SPOT = "选择景点"
    SELECT_ROUTE = "选择路线"


# ---------- 处理链 示例代码 ----------
def build_processing_chain(prompt: PromptTemplate) -> RunnableSequence:
    """处理链模版"""
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
                "content": seg.strip() + '  \n'  # Markdown换行
            }],
            "options": options or []
        }
        time.sleep(0.1)  # 更自然的流式速度

# ---------- Markdown渲染 未实装----------
def render_markdown(content: str) -> str:
    """安全渲染Markdown内容"""
    sanitized = re.sub(r'[^\w\s\-\.\!\?\u4e00-\u9fa5]', '', content)
    return f"```markdown\n{sanitized}\n```"


def create_llm():
    """创建LLM实例"""
    # NIM模版代码
    return ChatNVIDIA(
        model="meta/llama-3.3-70b-instruct",
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
        nvidia_api_key=NVIDIA_API_KEY
    )


# ---------- 新增链式状态处理 ----------
def build_state_chain() -> RunnableBranch:
    """构建链式状态处理器"""
    return RunnableBranch(
        # INIT状态处理
        (lambda s: s["step"] == TourismState.INIT,
         RunnableAssign({
             "messages": RunnableLambda(handle_init) | format_streaming_response,
             "options": RunnableLambda(lambda s: parse_options(s["messages"], "spot")),
             "step": lambda _: TourismState.SELECT_SPOT
         })),

        # GET_PREFERENCE状态处理 移除
        # (lambda s: s["step"] == TourismState.GET_PREFERENCE,
        #  RunnableAssign({
        #      "messages": RunnableLambda(handle_preference) | format_streaming_response,
        #      "options": RunnableLambda(lambda s: parse_options(s["messages"], "spot")),
        #      "step": lambda _: TourismState.SELECT_SPOT
        #  })),

        # SELECT_SPOT状态处理 
        (lambda s: s["step"] == TourismState.SELECT_SPOT,
         RunnableAssign({
             "messages": RunnableLambda(handle_spot_selection) | format_streaming_response,
             "options": RunnableLambda(lambda s: parse_options(s["messages"], "route")),
             "step": lambda _: TourismState.SELECT_ROUTE
         })),

        # 新增SELECT_ROUTE状态处理
        (lambda s: s["step"] == TourismState.SELECT_ROUTE,
         RunnableAssign({
             "messages": RunnableLambda(handle_route_selection) | format_streaming_response,
             "step": lambda _: TourismState.INIT
         })),

        # 默认处理 (添加True作为默认条件)
        (True,
         RunnableAssign({
             "messages": RunnableLambda(handle_unknown) | format_streaming_response,
             "step": lambda _: TourismState.INIT
         }))
    )


# ---------- 更新状态机函数 ----------
def state_machine(state: Dict, user_input: str) -> Generator[Dict, None, None]:
    """增强型状态机"""
    current_state = {
        "step": state.get("step", TourismState.INIT),
        # "preference": state.get("preference"),
        "selected_spot": state.get("selected_spot"),
        "selected_route": state.get("selected_route"),
        "options": state.get("options", []),
        "context": state.get("context", {})  # 新增上下文存储
    }
    handlers = {
        TourismState.INIT: handle_init,
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
def handle_init(state: Dict, user_input: str) -> Generator[Dict, None, Dict]:
    """初始化处理"""
    if not user_input.strip():
        welcome_msg = "🏖️ 欢迎使用智能旅游助手！请描述您的旅行偏好（例如：想看皇家建筑、喜欢自然风光等）"
        yield from format_streaming_response(welcome_msg)
        return {**state, "step": TourismState.INIT}

    # 原有GET_PREFERENCE处理逻辑
    image_info = get_rag_multi_model_context("")  # 示例使用空图片输入
    chain = handle_preference_chain()
    result = chain.invoke({
        "preference_info": user_input,
        "image_info": image_info,
    })

    yield from format_streaming_response(result)

    # 明确返回更新后的状态
    return {
        **state,
        "step": TourismState.SELECT_SPOT,
        "options": parse_options(result, "spot"),
        "context": {
            "preference_info": user_input,
            "image_info": image_info,
            "timestamp": time.strftime("%Y-%m-%d %H:%M")
        }
    }


class SpotOption(BaseModel):
    spots: List[str] = Field(description="景点列表，格式：景点名称:景点描述")


class RouteOption(BaseModel):
    routes: List[str] = Field(description="路线列表，格式：路线名称:路线特色")


# 新增解析提示模板
OPTION_PROMPTS = {
    "spot": PromptTemplate(
        input_variables=["content"],
        template="""从下面的选项中严格按格式提取景点列表：
{content}

要求：
1. 必须返回严格JSON格式，包含spots字段的数组
2. 每个元素格式："景点名称:景点描述"
3. 示例：{{"spots": ["故宫博物院:明清皇家宫殿", "天坛公园:皇家祭祀场所"]}}
4. 必须使用双引号，禁止换行符
5. 结果只返回json，不要做其他说明"""
    ),
    "route": PromptTemplate(
        input_variables=["content"],
        template="""从下面的路线推荐中严格按格式提取路线列表：
{content}

要求：
1. 必须返回严格JSON格式，包含routes字段的数组  
2. 每个元素格式："路线名称:路线特色"
3. 示例：{{"routes": ["文化之旅:深度皇家建筑讲解", "自然之旅:园林景观游览"]}}
4. 必须使用双引号，禁止换行符
5. 结果只返回json，不要做其他说明"""
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
# 创建基础解析器
base_spot_parser = PydanticOutputParser(pydantic_object=SpotOption)
base_route_parser = PydanticOutputParser(pydantic_object=RouteOption)
# 使用OutputFixingParser包裹基础解析器
spot_parser = OutputFixingParser.from_llm(parser=base_spot_parser, llm=option_parser_llm)
route_parser = OutputFixingParser.from_llm(parser=base_route_parser, llm=option_parser_llm)

option_parser = {
    "spot": (
            RunnableParallel(
                completion=OPTION_PROMPTS["spot"] | option_parser_llm,
                prompt_value=OPTION_PROMPTS["spot"]
            )
            | RunnableLambda(lambda x: spot_parser.parse(x["completion"].content))
            | RunnableLambda(lambda x: x.spots)
    ),
    "route": (
            RunnableParallel(
                completion=OPTION_PROMPTS["route"] | option_parser_llm,
                prompt_value=OPTION_PROMPTS["route"]
            )
            | RunnableLambda(lambda x: route_parser.parse(x["completion"].content))
            | RunnableLambda(lambda x: x.routes)
    )
}


# 修改解析函数
def parse_options(content: str, option_type: str) -> List[str]:
    """使用专用提示解析选项"""
    try:
        # 获取解析中间结果用于调试
        raw_output = option_parser[option_type].invoke({"content": content})
        return raw_output
    except Exception as e:
        print(f"⚠️解析失败")
        # 增强型正则匹配
        patterns = {
            "spot": r"\d+\.\s+([^：]+)",
            "route": r"\d+\.\s+([^：]+)"
        }
        return re.findall(patterns[option_type], content)


def get_rag_multi_model_context(query: str) -> str:
    '''根据用户输入使用RAG多模态得到信息'''

    query = '"./data/dataImg/0000215a37942b17.jpg"'
    preprocessor = img_preprocess.Preprocessor()
    result = (preprocessor.preprocess_chain | img_preprocess.print_chain | img_preprocess.search_chain).invoke(query)
    print(result.get("matching_location"))

    try:
        print("rag result" + result)
    except Exception as e:
        print(e)
    print("rag matching_location")
    print(result.get("matching_location"))
    return str(result.get("matching_location")[0])


def handle_preference_chain() -> RunnableSequence:
    """专用偏好处理链"""
    llm = create_llm()
    spot_prompt = PromptTemplate(
        input_variables=["preference_info", "image_info"],  # 确保输入变量匹配
        template="""作为资深旅行规划师，根据用户需求推荐3个景点：
    
    用户需求：{preference_info}
    同时根据用户上传的图片检索到了下面的地方：{image_info}

    请按以下格式推荐：
    1. [景点名称]：[50字特色描述]
    2. [景点名称]：[50字特色描述]
    3. [景点名称]：[50字特色描述]"""
    )
    return (
            RunnablePassthrough()
            | spot_prompt
            | llm.bind(stop=["\n\n"])
            | RunnableLambda(lambda x: x.content)
    )


def handle_preference(state: Dict, image_input: str = None, user_input: str = None) -> Generator[Dict, None, Dict]:
    """动态生成景点推荐（知识库增强版）"""
    # 从知识库动态获取上下文
    image_info = get_rag_multi_model_context(image_input)  # RAG多模态解析

    # 正确初始化处理链
    chain = handle_preference_chain()
    result = chain.invoke({
        "preference_info": user_input,
        "image_info": image_info,
    })  # 参数在此处传递
    
    
    # 类型安全检查
    if not isinstance(result, str):
        result = str(result)
    

    # 类型安全检查
    if not isinstance(result, str):
        result = str(result)
    
    # 流式输出处理
    yield from format_streaming_response(result)

    return {
        **state,
        "step": TourismState.SELECT_SPOT,
        "options": parse_options(result, "spot"),
        "context": {
            "preference_info": user_input,
            "image_info": image_info,
            "timestamp": time.strftime("%Y-%m-%d %H:%M")
        }
    }


def spot_selection_parse(user_input: str, options: list, context: dict) -> str:
    """使用小模型解析景点选择"""
    prompt = PromptTemplate(
        template="从用户输入中解析景点选择：\n选项列表：{options}\n用户输入：{user_input}\n返回[景点名称：特色描述]",
        input_variables=["options", "user_input"]
    )
    chain = prompt | option_parser_llm | StrOutputParser()
    return chain.invoke({"options": options, "user_input": user_input})


def handle_spot_selection_chain() -> RunnableSequence:
    """专用景点处理链"""
    llm = create_llm()
    route_prompt = PromptTemplate(
        input_variables=["preference_info", "spot_info"],
        template="""作为资深旅行规划师，根据用户需求和选定景点设计3条特色旅游路线:
    用户偏好：{preference_info}
    选定景点：{spot_info}

    请设计3条特色游览路线：
    1. [路线名称]：[路线特色与亮点]
    2. [路线名称]：[路线特色与亮点]
    3. [路线名称]：[路线特色与亮点]"""
    )
    return (
            RunnablePassthrough()
            | route_prompt
            | llm
            | StrOutputParser()
    )


def handle_spot_selection(state: Dict, user_input: str) -> Generator[Dict, None, Dict]:
    """动态生成路线推荐（增强版）"""
    selected_spot = spot_selection_parse(
        user_input,
        state["options"],
        context=state["context"]
    )

    result = handle_spot_selection_chain().invoke({
        "preference_info": state["context"]["preference_info"],
        "spot_info": selected_spot
    })

    # 流式输出处理
    yield from format_streaming_response(result)

    return {
        **state,
        "step": TourismState.SELECT_ROUTE,
        "options": parse_options(result, "route"),
        "context": {
            "preference_info": state["context"]["preference_info"],
            "image_info": state["context"]["image_info"],
            "spot_info": selected_spot,
            "timestamp": time.strftime("%Y-%m-%d %H:%M")
        }
    }


def route_selection_parse(user_input: str, options: list, context: dict) -> str:
    """使用小模型解析路线选择"""
    prompt = PromptTemplate(
        template="从用户输入解析路线选择：\n选项列表：{options}\n用户输入：{user_input}\n返回[路线名称：特色描述]",
        input_variables=["options", "user_input"]
    )
    chain = prompt | option_parser_llm | StrOutputParser()
    return chain.invoke({"options": options, "user_input": user_input})


def handle_route_selection_chain() -> RunnableSequence:
    """专用攻略生成链"""
    llm = create_llm()
    guide_prompt = PromptTemplate(
        input_variables=["preference_info", "spot_info", "route_info"],
        template="""作为资深旅行规划师，根据下面的信息，做一个深度旅游攻略生成：
        用户需求：{preference_info}
        选定景点：{spot_info}
        选择路线：{route_info}
    请包含以下内容，输出一份排版好的旅游攻略：
    ## 行程安排（精确到小时）
    ## 必玩项目（包含推荐星级）
    ## 餐饮推荐
    ## 预算估算
    ## 实用贴士"""
    )
    return (
            RunnablePassthrough()
            | guide_prompt
            | llm
            | StrOutputParser()
    )


def handle_route_selection(state: Dict, user_input: str) -> Generator[Dict, None, Dict]:
    """动态生成攻略（增强版）"""

    # 解析用户选择的景点
    selected_route = route_selection_parse(
        user_input,
        state["options"],
        context=state["context"]
    )

    # 你好
    chain = handle_route_selection_chain()
    result = chain.invoke({
        "preference_info": state["context"]["preference_info"],
        "spot_info": state["context"]["spot_info"],
        "route_info": selected_route
    })

    # 流式输出处理
    yield from format_streaming_response(result)
    return {
        **state,
        "step": TourismState.INIT,
        "context": {
            "preference_info": state["context"]["preference_info"],
            "spot_info": state["context"]["spot_info"],
            "image_info": state["context"]["image_info"],
            "route_info": selected_route,
            "timestamp": time.strftime("%Y-%m-%d %H:%M")
        }
    }


def handle_unknown(state: Dict, _: str) -> Generator[Dict, None, None]:
    """未知状态处理"""
    yield from format_streaming_response("⚠️ 系统遇到未知状态，正在重置...")
    return {**state, "step": TourismState.INIT}


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
        print("\n" + "-" * 50)
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
        print("\n" + "-" * 50)
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
        print("\n" + "-" * 50)
    except StopIteration as e:
        state = e.value if isinstance(e.value, dict) else state

    # 验证最终状态
    assert state["step"] == TourismState.INIT, "完成流程后应重置状态"
    assert "selected_route" in state, "应记录已选路线"
    print("✅ 路线选择测试通过")
    return state


# ---------- 新增链式状态机实现 ----------
def new_state_machine(state: Dict, user_input: str) -> Generator[Dict, None, None]:
    """链式状态机（新增实现）"""
    processing_chain = (
            RunnablePassthrough.assign(
                # 预处理用户输入
                processed_input=lambda x: x["user_input"].strip()
            )
            | build_state_chain()
            | RunnableAssign({
        "final_output": lambda x: x["messages"][-1]["content"]
    })
    )

    try:
        # 执行处理链
        result = processing_chain.invoke({
            **state,
            "user_input": user_input,
            "messages": []
        })

        # 流式输出处理
        yield from result["messages"]
        yield {"final_state": result}

    except Exception as e:
        yield from format_streaming_response(f"⚠️ 处理异常：{str(e)}")
        yield {"final_state": state}


# 旧的交互演示函数
def interactive_demo():
    """命令行交互演示"""
    print("🛫 旅游助手 控制台交互模式（输入exit退出）")
    state = {"step": TourismState.INIT}

    while True:
        try:
            # 根据状态提示输入
            if state["step"] == TourismState.INIT:
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
                print("-" * 50)
                print(full_response)
                print("-" * 50)
                state = {"step": TourismState.INIT}  # 重置状态

        except KeyboardInterrupt:
            print("\n\n👋 感谢使用，再见！")
            break


def test_option_parser():
    """测试选项解析器"""
    print("\n=== 测试选项解析器 ===")
    test_content = """系统：1. 故宫博物院：中国明清两代皇家宫殿建筑群，保存着丰富的历史文化遗产和珍贵文物。2. 天坛公园：北京著名的皇家祭祀场所，拥有精美的建筑和优美的自然环境。3. 颐和园：中国最大的皇家宫苑，拥有湖光山色和精美的建筑，是休闲和文化体验的理想之地。"""

    try:
        # 测试正式解析器
        print("\n🔧 使用正式解析器：")
        options = parse_options(test_content, "spot")
        print("解析结果：")
        for i, opt in enumerate(options, 1):
            print(f"  {i}. {opt}")

        # 测试备用解析方案
        print("\n🔄 触发备用正则方案：")
        raise Exception("模拟JSON解析失败")  # 强制触发备用方案
    except Exception as e:
        print(f"⚠️ 故意触发异常：{str(e)}")
        options = re.findall(r"\d+\.\s+\*\*(.*?)\*\*", test_content)
        print("备用方案结果：")
        for i, opt in enumerate(options, 1):
            print(f"  {i}. {opt}")

    print("\n✅ 解析测试完成")


langchain.debug = False
# ---------- 更新主流程 ----------
if __name__ == "__main__":
    print("开始咯~")
    # 初始化大模型
    llm = create_llm()
    print("✅ 大模型初始化成功")
    # test_option_parser()
    # 启动交互模式
    # interactive_demo_with_chain() 使用状态链的demo，需要DEBUG
    interactive_demo()  # 旧状态机函数
    # get_rag_multi_model_context("")

