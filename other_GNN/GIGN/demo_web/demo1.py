import gradio as gr

# 定义简单的函数，返回用户输入的文本
def echo_text(input_text):
    return f'You entered: {input_text}'

# 创建 Gradio 界面
demo = gr.Interface(fn=echo_text,  # 绑定的函数
                    inputs="text",  # 输入类型：文本
                    outputs="text",  # 输出类型：文本
                    title="Simple Gradio Webpage,you did it. come on,come on",  # 网页标题
                    description="damn it")  # 网页描述

# 启动 Gradio 应用
demo.launch(server_name="0.0.0.0", server_port=5576,share=False)


