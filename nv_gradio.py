#coding:utf-8
import gradio as gr
from PIL import Image

import img_preprocess
import tourism_assistant

# 模拟一个简单的 AI 回答函数，支持流式输出

final_state = {}
def chart_agent_gr(image_input, user_input, chat_history):
    global final_state

    full_response = ""

    # 更新状态判断逻辑（移除GET_PREFERENCE）
    if final_state.get("step") == tourism_assistant.TourismState.INIT:
        if not image_input and not user_input:
            full_response = "🏖️ 欢迎使用智能旅游助手！请描述旅行偏好或上传景点图片"
            chat_history.append((user_input, full_response))
            yield chat_history
            return
        else:
            # 合并处理文本和图片输入
            response_gen = tourism_assistant.state_machine(final_state, user_input or image_input)
    else:
        response_gen = tourism_assistant.state_machine(final_state, user_input)
    try:
        temp_state = {}
        current_response = ""
        for chunk in response_gen:
            if chunk.get("final_state"):
                temp_state = chunk["final_state"]
                continue
            content = chunk['messages'][0]['content']
            current_response += content
            # 更新历史记录判断条件
            updated_history = chat_history + [
                (user_input if final_state.get("step") != tourism_assistant.TourismState.INIT 
                 else f'<img src="data:image/png;base64,{img_preprocess.convert_image_to_base64(image_input)}" alt="image" style="max-width: 300px; max-height: 300px;">', 
                 current_response)
            ]
            yield updated_history
        # 更新最终状态处理
        chat_history.append(
            (user_input if final_state.get("step") != tourism_assistant.TourismState.INIT
             else f'<img src="data:image/png;base64,{img_preprocess.convert_image_to_base64(image_input)}" alt="image" style="max-width: 300px; max-height: 300px;">', 
             current_response)
        )
        final_state = temp_state
    except StopIteration:
        pass

# 创建 Gradio 界面
with gr.Blocks() as demo:
    demo.css = """
    .wrap-long-text {
        white-space: pre-wrap;
        word-break: break-word;
        line-height: 1.6;  /* 增加行高 */
        max-width: 800px;  /* 限制最大宽度 */
    }
    .wrap-long-text p {
        margin: 0.5em 0;  /* 段落间距 */
    }
    """
    # 创建一个聊天界面
    chatbot = gr.Chatbot(label="Chat History", render=gr.Markdown(),elem_classes=["wrap-long-text"])
    with gr.Row():
        text_input = gr.Textbox(placeholder="Type your message here...", label="Text Input")
        image_input = gr.Image(label="Image Input", type="filepath")  # 获取图片路径

    # 创建一个提交按钮
    submit_button = gr.Button("Send")

    # 使用 state 保存对话历史
    chat_state = gr.State([])  # 初始化对话历史


    # 定义提交按钮的回调函数
    def respond(text_input, image_input, chat_history):
        # 如果是文本输入，调用 chart_agent_gr 并流式更新聊天历史
        for updated_history in chart_agent_gr(image_input, text_input, chat_history):
            yield "", None, updated_history


    # 绑定回调函数
    submit_button.click(
        respond,
        inputs=[text_input, image_input, chat_state],
        outputs=[text_input, image_input, chatbot],
    )


    # 清空聊天历史
    def clear_chat():
        return []


    clear_button = gr.Button("Clear History")
    clear_button.click(clear_chat, outputs=chat_state)

# 启动 Gradio 应用
demo.launch(debug=True, share=False, server_port=5000, server_name="0.0.0.0",)
