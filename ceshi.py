import gradio as gr
from PIL import Image
import os


# 模拟一个简单的 AI 回答函数
def ai_response(message, history):
    # 这里可以替换为实际的 AI 模型
    if isinstance(message, str):
        return f"AI: You said '{message}'"
    elif isinstance(message, str) and message.endswith((".png", ".jpg", ".jpeg", ".gif")):
        # 如果是图片路径，返回 Markdown 格式的图片标签
        return f"AI: You sent an image! Here it is: ![image]({message})"
    else:
        return "AI: I don't understand that input."


# 创建 Gradio 界面
with gr.Blocks() as demo:
    # 创建一个聊天界面
    chatbot = gr.Chatbot(label="Chat History")
    with gr.Row():
        text_input = gr.Textbox(placeholder="Type your message here...", label="Text Input")
        image_input = gr.Image(label="Image Input", type="filepath")  # 获取图片路径

    # 创建一个提交按钮
    submit_button = gr.Button("Send")


    # 定义提交按钮的回调函数
    def respond(message, image, chat_history):
        if message:
            response = ai_response(message, chat_history)
            chat_history.append((message, response))
        elif image:
            # 获取图片路径
            image_path = image
            print(f"Image saved at: {image_path}")  # 打印图片路径（后台处理）

            # 示例：使用 PIL 打开图片
            with Image.open(image_path) as img:
                print(f"Image size: {img.size}")  # 打印图片尺寸

            # 将图片路径转换为 Markdown 格式
            image_markdown = f"![image]({image_path})"
            response = ai_response(image_path, chat_history)
            chat_history.append((image_markdown, response))
        return "", None, chat_history


    # 绑定回调函数
    submit_button.click(respond, inputs=[text_input, image_input, chatbot], outputs=[text_input, image_input, chatbot])

# 启动 Gradio 应用
demo.launch()