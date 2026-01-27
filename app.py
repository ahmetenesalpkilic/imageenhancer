import gradio as gr
import numpy as np

def upscale_image(input_img):
    # input_img: RGB NumPy array [H, W, 3]
    # Şimdilik test amaçlı aynısını döndürelim
    ai_output_img = input_img
    return ai_output_img

demo = gr.Interface(
    fn=upscale_image,
    inputs=gr.Image(type="numpy", label="Düşük Çözünürlüklü Resim"),
    outputs=gr.Image(type="numpy", label="AI Tarafından İyileştirilmiş Resim"),
    title="AI Image Enhancer (SRCNN)"
)

demo.launch()
