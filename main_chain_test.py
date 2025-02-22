# %!pip install websockets
# %pip install langchain
# %pip install langchain_nvidia_ai_endpoints
# %pip install langchain_community gradio
# %pip install faiss-cpu
# %pip install pyppeteer
# %pip install markupsafe

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
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

# 初始化LLM
llm = ChatNVIDIA(
    model="deepseek-ai/deepseek-r1",
    temperature=0.6,
    top_p=0.7,
    max_tokens=4096,
    nvidia_api_key="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"
)

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

def parse_options(text):
    """解析带编号的选项"""
    return re.findall(r"\d+\.\s+(.*?):", text)

def format_response(message, options=None, markup=False):
    """构建标准化响应"""
    response = {"message": message}
    if options:
        response["options"] = options
    if markup:
        response["message"] = f"```markdown\n{message}\n```"
    return response

async def get_ai_designed_css(md_content):
    """调用大模型生成CSS样式"""
    css_llm = ChatNVIDIA(
        model="deepseek-ai/deepseek-r1",
        temperature=0.7,
        nvidia_api_key="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"
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
    
    response = await css_llm.agenerate([prompt])
    css = response.generations[0].text
    return css

async def render_md_to_image(md_content, output_path):
    """将Markdown渲染为图片"""
    css = await get_ai_designed_css(md_content)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            {css}
            /* 基础样式 */
            body {{
                margin: 0;
                padding: 40px;
                width: 1080px;
                min-height: 1920px;
            }}
            /* Markdown基础样式 */
            h1 {{ margin-bottom: 1em; }}
            p {{ line-height: 1.6; }}
        </style>
    </head>
    <body>
        <div class="poster-container">
            {Markup(md_content)}
        </div>
    </body>
    </html>
    """
    
    try:
        browser = await launch({
            'headless': True,
            'args': ['--no-sandbox', '--disable-setuid-sandbox']
        })
        page = await browser.newPage()
        await page.setViewport({"width": 1080, "height": 1920})
        await page.setContent(html)
        
        # 使用当前脚本所在目录作为基准路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        full_output_path = os.path.join(current_dir, output_path)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
        
        await page.screenshot({
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
            await browser.close()

async def process_step(state, user_input):
    """核心处理逻辑"""
    new_state = state.copy()
    
    try:
        # 初始状态
        if state["step"] == State.INIT:
            new_state = {
                "step": State.GET_PREFERENCE,
                "preference": None,
                "spot": None,
                "route": None
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
                | llm 
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
            
            route_chain = (
                RunnablePassthrough()
                | route_prompt
                | llm
                | RunnableLambda(lambda x: x.content)
            )
            
            result = route_chain.invoke({
                "preference": state.get("preference", ""),
                "spot": user_input
            })
            
            routes = parse_options(result)
            new_state["step"] = State.SELECT_ROUTE
            return new_state, format_response(result, routes)
            
        # 攻略生成步骤
        elif state["step"] == State.SELECT_ROUTE:
            new_state["route"] = user_input
            
            guide_chain = (
                RunnablePassthrough()
                | guide_prompt
                | llm
                | RunnableLambda(lambda x: x.content)
            )
            
            result = guide_chain.invoke({
                "preference": state.get("preference", ""),
                "spot": state.get("spot", ""),
                "route": user_input
            })

            # 提取markdown内容
            md_content = result.strip("```markdown\n").strip("```")
            
            # 生成图片文件名
            output_path = f"travel_guide_{int(time.time())}.jpg"
            
            # 渲染为图片
            await render_md_to_image(md_content, output_path)
            
            new_state["step"] = State.INIT
            return new_state, format_response(
                f"攻略已生成!\n\n{result}\n\n图片已保存至: {output_path}", 
                markup=True
            )
            
    except Exception as e:
        print(f"Error details: {e}")
        return state, format_response(f"⚠️ 出错：{str(e)}，请重新输入")
    
    return state, format_response("未知状态")

# 测试代码
async def test_travel_assistant():
    print("开始测试...")
    
    # 初始化状态
    state = {"step": State.INIT}
    print("初始状态:", state)
    
    # 测试初始状态
    state, response = await process_step(state, "")
    print("初始响应:", response)
    print("更新后状态:", state)
    print("等待15秒...")
    for i in range(15, 0, -1):
        print(f"\r倒计时: {i}秒", end="", flush=True)
        await asyncio.sleep(1)
    print("\n等待完成")

    # 测试获取偏好
    print("\n测试获取偏好...")
    state, response = await process_step(state, "我想去一个历史文化景点，最好能体验古代宫廷文化")
    print("偏好响应:", response)
    print("更新后状态:", state)
    print("等待15秒...")
    for i in range(15, 0, -1):
        print(f"\r倒计时: {i}秒", end="", flush=True)
        await asyncio.sleep(1)
    print("\n等待完成")

    # 测试选择景点
    print("\n测试选择景点...")
    state, response = await process_step(state, "北京故宫")
    print("景点响应:", response)
    print("更新后状态:", state)
    print("等待15秒...")
    for i in range(15, 0, -1):
        print(f"\r倒计时: {i}秒", end="", flush=True)
        await asyncio.sleep(1)
    print("\n等待完成")

    # 测试选择路线
    print("\n测试选择路线...")
    state, response = await process_step(state, "故宫深度一日游")
    print("路线响应:", response)
    print("更新后状态:", state)
if __name__ == "__main__":
    asyncio.run(test_travel_assistant())
