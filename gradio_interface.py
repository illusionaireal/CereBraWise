import gradio as gr
from tourism_assistant import *
import time

# 状态样式表
CSS = """
.important-text {color: #2B5876; font-weight: bold;}
.option-list {border: 1px solid #E0E0E0; border-radius: 5px; padding: 15px; margin: 10px 0;}
"""

def update_ui_state(current_step: TourismState):
    """根据当前步骤更新UI元素可见性"""
    return {
        init_components: gr.update(visible=current_step == TourismState.INIT),
        spot_components: gr.update(visible=current_step == TourismState.SELECT_SPOT),
        route_components: gr.update(visible=current_step == TourismState.SELECT_ROUTE),
        chat_window: gr.update(visible=True)
    }

def process_input(state: dict, user_input: str, chat_history: list, file=None):
    """处理用户输入的核心逻辑"""
    # 处理图片上传
    if file:
        # 使用安全文件名替换中文标点
        safe_filename = file.name.replace("：", "_")  # 替换中文冒号
        state["image_path"] = safe_filename
        user_input += f" [已上传图片：{safe_filename}]"  # 显示时仍保留中文冒号
    
    # 执行状态机
    response_gen = state_machine(state, user_input)
    
    # 处理流式响应
    full_response = ""
    for chunk in response_gen:
        if "final_state" in chunk:
            state = chunk["final_state"]
        elif "messages" in chunk:
            content = chunk['messages'][0]['content']
            full_response += content
            chat_history.append((user_input, full_response))
            yield chat_history, state, ""
            
            # 模拟逐字输出效果
            time.sleep(0.1)
            chat_history[-1] = (user_input, full_response)
    
    # 更新选项显示
    if state["step"] == TourismState.SELECT_SPOT and state.get("options"):
        options = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(state["options"])])
        chat_history.append(("", f"**请选择景点：**\n<div class='option-list'>{options}</div>"))
    elif state["step"] == TourismState.SELECT_ROUTE and state.get("options"):
        options = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(state["options"])])
        chat_history.append(("", f"**请选择路线：**\n<div class='option-list'>{options}</div>"))

def handle_image_upload(file):
    """处理图片上传"""
    # 移除所有可能引起路径问题的特殊字符
    safe_filename = file.name.replace("：", "_").replace(":", "_").replace(" ", "_")
    # 返回显示信息时完全移除冒号
    return f"已上传图片_{safe_filename}"

with gr.Blocks(css=CSS) as demo:
    state = gr.State({"step": TourismState.INIT})
    
    with gr.Column(visible=True, elem_id="init_components") as init_components:
        gr.Markdown("## 🏯 智能旅游规划助手")
        image_upload = gr.UploadButton("📤 上传参考图片", file_types=["image"])
        image_preview = gr.Image(label="上传的图片")
        init_prompt = gr.Markdown("✨ 请描述您的旅游偏好（例如：想看皇家建筑/喜欢自然风光）：")
    
    with gr.Column(visible=False, elem_id="spot_components") as spot_components:
        gr.Markdown("### 请从以下景点中选择", elem_classes="important-text")
    
    with gr.Column(visible=False, elem_id="route_components") as route_components:
        gr.Markdown("### 请从以下路线中选择", elem_classes="important-text")
    
    with gr.Column(visible=True) as chat_window:
        chatbot = gr.Chatbot(height=500)
        msg = gr.Textbox(label="输入消息")
        submit_btn = gr.Button("发送", variant="primary")
    
    # 事件绑定
    image_upload.upload(
        handle_image_upload, 
        inputs=image_upload,
        outputs=image_preview
    )
    
    submit_btn.click(
        process_input,
        [state, msg, chatbot, image_upload],
        [chatbot, state, msg]
    ).then(
        update_ui_state,
        inputs=state,
        outputs=[init_components, spot_components, route_components, chat_window]
    )
    
    msg.submit(
        process_input,
        [state, msg, chatbot, image_upload],
        [chatbot, state, msg]
    ).then(
        update_ui_state,
        inputs=state,
        outputs=[init_components, spot_components, route_components, chat_window]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860) 