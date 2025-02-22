import gradio as gr
from PIL import Image
import tourism_assistant

# 模拟一个简单的 AI 回答函数，支持流式输出

final_state = {}
def chart_agent_gr(image_input, user_input, chat_history):
    global final_state

    full_response = ""
    if final_state.get("step") == tourism_assistant.TourismState.GET_PREFERENCE:
        if not image_input:
            full_response = "您喜欢什么样的景点呢？是自然风光、历史文化，还是城市地标？如果您有相关图片，上传给我们，我们会为您量身定制推荐！"
            chat_history.append((user_input, full_response))
            yield chat_history  # 流式返回更新后的聊天历史
            return
        else:
            response_gen = tourism_assistant.state_machine(final_state, image_input)
    else:
        response_gen = tourism_assistant.state_machine(final_state, user_input)
    try:
        current_response = ""
        for chunk in response_gen:
            if chunk.get("final_state"):
                final_state = chunk["final_state"]
                continue
            content = chunk['messages'][0]['content']
            current_response += content
            # 合并历史记录和当前生成的内容
            updated_history = chat_history + [(user_input if (not image_input) and final_state.get("step") != tourism_assistant.TourismState.GET_PREFERENCE else f"![image]({image_input})", current_response)]
            yield updated_history  # 流式返回更新后的聊天历史
        # 生成完成后，将完整的对话追加到历史记录中
        chat_history.append((user_input if (not image_input) and final_state.get("step") != tourism_assistant.TourismState.GET_PREFERENCE else f"![image]({image_input})", current_response))
        print("\n" + "-" * 50)
    except StopIteration:
        pass
    #         full_response += content
    #         # 更新聊天历史并实时返回
    #         chat_history.append((user_input if (not image_input) and final_state.get("step") != tourism_assistant.TourismState.GET_PREFERENCE else f"![image]({image_input})" , full_response))
    #         yield chat_history  # 流式返回更新后的聊天历史
    #     print("\n" + "-" * 50)
    # except StopIteration:
    #     pass

# 创建 Gradio 界面
with gr.Blocks() as demo:
    # 创建一个聊天界面
    chatbot = gr.Chatbot(label="Chat History")
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
demo.launch(debug=True, share=False, server_port=5000, server_name="127.0.0.1")