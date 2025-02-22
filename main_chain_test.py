# %!pip install websockets
# %pip install langchain
# %pip install langchain_nvidia_ai_endpoints
# %pip install langchain_community gradio
# %pip install faiss-cpu
# %pip install pyppeteer
# %pip install markupsafe

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
import gradio as gr
from enum import Enum
from langchain_community.vectorstores import FAISS
import re
import time
import asyncio
import os
from markupsafe import Markup
from pyppeteer import launch

# 初始化向量存储
vector_store = None


def init_vector_store():
    global vector_store
    embeddings = NVIDIAEmbeddings(
        model="NV-Embed-QA",
        nvidia_api_key="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"
    )

    documents = [
        "北京故宫是中国明清两代的皇家宫殿",
        "上海外滩是著名的历史建筑群",
        "西安兵马俑是秦始皇的陪葬坑",
        "杭州西湖有十景包括苏堤春晓等",
        "广州塔昵称小蛮腰，高600米"
    ]

    vector_store = FAISS.from_texts(
        texts=documents,
        embedding=embeddings,
        metadatas=[{"source": "knowledge"}] * len(documents)
    )
    return vector_store


# 状态枚举优化
class State(Enum):
    INIT = 0
    GET_PREFERENCE = 1
    SELECT_SPOT = 2
    SELECT_ROUTE = 3
    SHOW_GUIDE = 4


# ========== 初始化LLM ==========
llm = ChatNVIDIA(
    model="deepseek-ai/deepseek-r1",
    temperature=0.6,
    top_p=0.7,
    max_tokens=4096,
    nvidia_api_key="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"
)


# 新增解析器组件
class SmartOptionParser:
    def __init__(self):
        self.parser_llm = ChatNVIDIA(
            model="mistralai/mistral-7b-instruct-v0.2",
            temperature=0.1,
            max_tokens=20,
            nvidia_api_key="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"
        )
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "根据用户输入解析出选项编号（数字）。只需返回数字，不要解释。\n选项列表：{options}"),
            ("human", "{input}")
        ])
        self.chain = self.prompt_template | self.parser_llm | StrOutputParser()

    def parse_async(self, user_input: str, options: list) -> int:
        try:
            response = self.chain.ainvoke({
                "input": user_input,
                "options": "\n".join([f"{i + 1}. {opt}" for i, opt in enumerate(options)])
            })

            # 处理中文数字
            chinese_num_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}
            if response.strip() in chinese_num_map:
                return chinese_num_map[response.strip()]

            return int(re.search(r"\d+", response).group())
        except Exception as e:
            print(f"智能解析失败: {str(e)}, 原始响应: {response}")
            raise ValueError("无法理解您的选择，请使用数字编号选择（如：1、2、3）")
# 初始化解析器
option_parser = SmartOptionParser()


# 优化提示模板
spot_prompt = PromptTemplate(
    input_variables=["context", "preference"],
    template="""作为旅行规划专家，根据以下信息推荐3个景点：

背景知识：
{context}

用户偏好：{preference}

要求：
1. 按编号列出选项，格式：
   1. [景点名称]：[50字简介]
   2. [景点名称]：[50字简介]
   3. [景点名称]：[50字简介]
2. 每个景点用1行描述"""
)

route_prompt = PromptTemplate(
    input_variables=["preference", "spot"],
    template="""作为一位专业的旅游规划师，请根据以下信息设计游玩路线：

用户偏好：{preference}
选择的景点：{spot}

请设计3种不同特色的游玩路线：

输出要求：
1. 严格按照以下格式输出三个路线：
   1. [路线名称]：[特色描述，包含游玩时间、特色体验等]
   2. [路线名称]：[特色描述，包含游玩时间、特色体验等]
   3. [路线名称]：[特色描述，包含游玩时间、特色体验等]
2. 每个路线必须包含具体的时间安排
3. 每个路线描述控制在80字以内
4. 确保输出格式规范，不要有多余的标点符号

示例输出：
1. 文化探索路线：时长1天，上午参观太和殿，下午游览珍宝馆，体验皇家文化
2. 摄影精选路线：时长2天半，清晨拍摄金光万字廊，午后取景御花园，记录宫廷之美
3. 深度体验路线：时长3天，专业讲解太和殿，午餐御膳房，下午欣赏宫廷文物"""
)

guide_prompt = PromptTemplate(
    input_variables=["preference", "spot", "route"],
    template="""用户偏好：{preference}
选择的景点：{spot}
选择的路线：{route}

请创建详细攻略：
包含：
- 行程安排（时间表）
- 必玩项目
- 美食推荐
- 住宿建议
- 预算估算
输出格式：Markdown"""
)

def generate_guide(state):
    """生成旅游攻略"""
    try:
        guide_chain = (
            RunnablePassthrough()
            | guide_prompt
            | llm.stream
            | RunnableLambda(lambda x: x.content)
        )

        result = guide_chain.invoke({
            "preference": state.get("preference", ""),
            "spot": state.get("spot", ""),
            "route": state.get("route", "")
        })

        # 提取markdown内容
        md_content = result.strip("```markdown\n").strip("```")

        # 生成图片文件名
        output_path = f"travel_guide_{int(time.time())}.jpg"

        # 渲染为图片
        render_md_to_image(md_content, output_path)

        new_state = state.copy()
        new_state["step"] = State.INIT
        return new_state, format_response(
            f"攻略已生成!\n\n{result}\n\n图片已保存至: {output_path}",
            markup=True
        )
    except Exception as e:
        print(f"攻略生成错误: {str(e)}")
        return state, format_response(f"⚠️ 攻略生成失败：{str(e)}，请重试")
        
# 修改parse_options函数增强容错性
def parse_options(text):
    """解析带编号的选项（增强版）"""
    # 扩展匹配模式，支持中文编号和不同分隔符
    return re.findall(r"(?:\d+|[一二三四五六七八九十]+)\.\s+(.*?)[：:]", text)



def format_response(message, options=None, markup=False):
    """构建标准化响应（符合官方消息格式）"""
    response = {
        "messages": [{
            "role": "assistant", 
            "content": f"```markdown\n{message}\n```" if markup else message
        }]
    }
    if options:
        response["options"] = [{"label": opt, "value": str(i+1)} for i, opt in enumerate(options)]
    return response


def get_ai_designed_css(md_content):
    """调用大模型生成CSS样式"""
    css_llm = ChatNVIDIA(
        model="deepseek-ai/deepseek-r1",
        temperature=0.7,
        nvidia_api_key=API_KEY
    )

    prompt = f"""作为专业平面设计师,请为以下Markdown内容设计海报样式:
    {md_content}
    
    要求:
    1. 使用优雅的渐变背景
    2. 合理的字体层级和间距
    3. 重要内容突出显示
    4. 适合1080x1920的手机海报尺寸
    
    请直接返回CSS代码(不要解释)。
    """

    response = css_llm.agenerate([prompt])
    css = response.generations[0].text
    return css


def render_md_to_image(md_content, output_path):
    """将Markdown渲染为图片"""
    try:
        # 加强内容过滤（修复中文错误关键点）
        md_content = re.sub(r'[^\x00-\x7F\u4e00-\u9fa5\s\.\,\!\?\-\:\/\（\）\《\》]', '', md_content)

        # 获取CSS时添加严格过滤
        css = get_ai_designed_css(md_content)
        css = re.findall(r'/\*.*?\*/|{[^{}]*}|.*?{.*?}', css, re.DOTALL)[0]  # 提取有效CSS部分

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                {css}
                /* 强制基础样式 */
                body {{
                    margin: 0 !important;
                    padding: 40px !important;
                    width: 1080px !important;
                    min-height: 1920px !important;
                    font-family: "Microsoft YaHei", sans-serif !important;
                }}
                h1 {{ margin-bottom: 1em !important; }}
                p {{ line-height: 1.6 !important; }}
            </style>
        </head>
        <body>
            <div class="poster-container">
                {Markup(md_content)}
            </div>
        </body>
        </html>
        """

        # 使用更可靠的文件路径处理
        current_dir = os.path.dirname(os.path.abspath(__file__))
        full_output_path = os.path.join(current_dir, output_path.lstrip('./'))

        browser =  launch({
            'headless': True,
            'args': ['--no-sandbox', '--disable-setuid-sandbox']
        })
        page = browser.newPage()
        page.setViewport({"width": 1080, "height": 1920})
        page.setContent(html, waitUntil='networkidle0')  # 增加加载等待

        page.screenshot({
            "path": full_output_path,
            "fullPage": True,
            "type": "jpeg",
            "quality": 90
        })
    except Exception as e:
        print(f"图片生成错误: {str(e)}")
        raise e
    finally:
        if 'browser' in locals():
            browser.close()


def process_step(state, user_input):
    """核心处理逻辑"""
    new_state = state.copy()

    try:
        # 初始状态
        if state["step"] == State.INIT:
            new_state = {
                "step": State.GET_PREFERENCE,
                "preference": None,
                "spot": None,
                "route": None,
                "last_user_input": None
            }
            return new_state, format_response("🏖️ 请问您想要什么样的旅游呢？")

        # 获取旅游偏好
        elif state["step"] == State.GET_PREFERENCE:
            new_state["preference"] = user_input

            # 使用知识库增强用户输入
            if vector_store is None:
                init_vector_store()
            docs = vector_store.similarity_search(user_input, k=3)
            context = "\n".join([d.page_content for d in docs])

            # 构建正确的输入字典
            spot_chain = (
                    RunnablePassthrough()
                    | spot_prompt
                    | llm.stream
                    | RunnableLambda(lambda x: x.content)
            )

            result = spot_chain.invoke({
                "context": context,
                "preference": user_input
            })

            spots = parse_options(result)
            new_state["step"] = State.SELECT_SPOT
            return new_state, format_response(result, spots)

        # 路线推荐步骤
        elif state["step"] == State.SELECT_SPOT:
            new_state["spot"] = user_input
            new_state["last_user_input"] = user_input

            route_chain = (
                    RunnablePassthrough()
                    | route_prompt
                    | llm.stream
                    | RunnableLambda(lambda x: x.content)
            )

            try:
                result = route_chain.ainvoke({
                    "preference": state.get("preference", ""),
                    "spot": user_input
                })

                print(f"【DEBUG】路线推荐原始输出:\n{result}")
                routes = parse_options(result)  # 保留原始解析作为fallback

                # 智能解析用户选择
                selected_index =  option_parser.parse_async(
                    user_input=state["last_user_input"],  # 需要记录用户原始输入
                    options=routes
                )

                if selected_index and routes:
                    new_state["route"] = routes[selected_index - 1]
                    new_state["step"] = State.SELECT_ROUTE
                    return new_state, format_response(result, routes)
            except Exception as e:
                print(f"路线推荐错误: {str(e)}")
                return state, format_response(f"⚠️ 路线生成失败：{str(e)}，请重新选择")

        # 攻略生成步骤
        elif state["step"] == State.SELECT_ROUTE:
            new_state["route"] = user_input

            print(f"User input route: {user_input}")  # 调试输出

            guide_chain = (
                    RunnablePassthrough()
                    | guide_prompt
                    | llm.stream
                    | RunnableLambda(lambda x: x.content)
            )

            print("Guide chain created.")  # 调试输出

            result = guide_chain.invoke({
                "preference": state.get("preference", ""),
                "spot": state.get("spot", ""),
                "route": user_input
            })

            print(f"Raw result from guide_chain: {result}")  # 调试输出

            # 提取markdown内容
            md_content = result.strip("```markdown\n").strip("```")
            print(f"Extracted markdown content: {md_content}")  # 调试输出

            # 生成图片文件名
            output_path = f"travel_guide_{int(time.time())}.jpg"
            print(f"Output image path: {output_path}")  # 调试输出

            # 渲染为图片
            render_md_to_image(md_content, output_path)
            print(f"Image rendering completed.")  # 调试输出

            new_state["step"] = State.INIT
            return new_state, format_response(
                f"攻略已生成!\n\n{result}\n\n图片已保存至: {output_path}",
                markup=True
            )

    except Exception as e:
        print(f"Error details: {e}")
        return state, format_response(f"⚠️ ERROR：{str(e)}，请重新输入")

    return state, format_response("未知状态")


# 测试代码
def test_travel_assistant():
    print("开始测试...")

    # 初始化状态
    state = {"step": State.INIT}
    print("初始状态:", state)

    # 测试初始状态
    new_state, response = process_step(state, "")
    print("初始响应:", response)
    print("更新后状态:", new_state)
    print("等待15秒...")
    for i in range(15, 0, -1):
        print(f"\r倒计时: {i}秒", end="", flush=True)
        time.sleep(1)
    print("\n等待完成")

    # 测试获取偏好
    print("\n测试获取偏好...")
    user_input = "我想去一个历史文化景点，最好能体验古代宫廷文化"
    new_state, response_stream = process_step(new_state, user_input)
    print("偏好响应: ", end="", flush=True)
    
    # 处理流式输出
    response_text = ""
    for chunk in response_stream:
        if isinstance(chunk, str):
            response_text += chunk
            print(chunk, end="", flush=True)
    print("\n更新后状态:", new_state)
    
    print("等待15秒...")
    for i in range(15, 0, -1):
        print(f"\r倒计时: {i}秒", end="", flush=True)
        time.sleep(1)
    print("\n等待完成")

    # 测试选择景点
    print("\n测试选择景点...")
    user_input = "我选择第2个景点"
    new_state, response_stream = process_step(new_state, user_input)
    print("景点响应: ", end="", flush=True)
    
    # 处理流式输出
    response_text = ""
    for chunk in response_stream:
        if isinstance(chunk, str):
            response_text += chunk
            print(chunk, end="", flush=True)
    print("\n更新后状态:", new_state)
    
    print("等待15秒...")
    for i in range(15, 0, -1):
        print(f"\r倒计时: {i}秒", end="", flush=True)
        time.sleep(1)
    print("\n等待完成")

    # 测试选择路线
    print("\n测试选择路线...")
    user_input = "我选择第三条路线"
    new_state, response_stream = process_step(new_state, user_input)
    print("路线响应: ", end="", flush=True)
    
    # 处理流式输出
    response_text = ""
    for chunk in response_stream:
        if isinstance(chunk, str):
            response_text += chunk
            print(chunk, end="", flush=True)
    print("\n更新后状态:", new_state)

if __name__ == "__main__":
    test_travel_assistant()

# def handle_interaction(user_input, history, current_state):
#     try:
#         new_state = current_state.copy()
#
#         # 获取流式响应
#         response_stream = process_step(current_state, user_input)
#
#         # 构建完整响应
#         full_response = ""
#         for chunk in response_stream:
#             if isinstance(chunk, dict):
#                 message = chunk.get("message", "")
#                 full_response += message
#                 # 实时更新聊天界面
#                 yield (
#                     history + [(user_input, full_response)],
#                     "",
#                     gr.update(visible=bool(chunk.get("options"))),
#                     gr.update(visible=not bool(chunk.get("options"))),
#                     new_state
#                 )
#             else:
#                 full_response += chunk
#                 yield (
#                     history + [(user_input, full_response)],
#                     "",
#                     gr.update(visible=False),
#                     gr.update(visible=True),
#                     new_state
#                 )
#
#     except Exception as e:
#         yield (
#             history + [(user_input, f"⚠️ 错误：{str(e)}，请重试")],
#             "",
#             gr.update(visible=False),
#             gr.update(visible=True),
#             current_state
#         )

        # # 修改事件绑定为流式输出
        # input_box.submit(
        #     handle_interaction,
        #     inputs=[input_box, chatbot, state],
        #     outputs=[
        #         chatbot,
        #         input_box,
        #         choices,
        #         input_box,
        #         state
        #     ],
        #     queue=True  # 启用队列以支持流式输出
        # ).then(
        #     lambda: None,
        #     None,
        #     [input_box],
        #     queue=False
        # )
