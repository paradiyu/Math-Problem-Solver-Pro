import io
import os
from google import genai
from google.genai import types
from PIL import Image

API_KEY = os.getenv("WHATA_API_KEY")
BASE_URL = "https://api.whatai.cc"
MODEL_TEXT = 'gemini-3-pro-preview' 
MODEL_IMAGE = 'gemini-3.1-flash-image-preview' 

def get_compressed_image_bytes(path):
    """读取并压缩图片，返回字节流"""
    with Image.open(path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.thumbnail((1600, 1600))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=85)
        return img_byte_arr.getvalue()

def process_pipeline(image_path):
    client = genai.Client(
        http_options=types.HttpOptions(base_url=BASE_URL),
        api_key=API_KEY
    )

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    md_filename = f"{base_name}_answer.md"
    final_image_name = f"{base_name}-answer.png"

    try:
        # --- 阶段一：生成文本解答 ---
        print(f"步骤 1/2: 正在提取题目并生成文本解答...")
        image_bytes = get_compressed_image_bytes(image_path)
        
        text_response = client.models.generate_content(
            model=MODEL_TEXT,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
                '解决这个问题，要求数学过程严格，使用严谨的数学语言，解答的时候将题目抄一遍，使用中文解答。'
            ],
            config=types.GenerateContentConfig(
                http_options=types.HttpOptions(
                    timeout=300000 # 5 分钟超时，绘图通常较慢
                )
            )
        )
        
        answer_text = text_response.text
        if not answer_text:
            raise Exception("模型未能生成文本解答内容。")

        with open(md_filename, "w", encoding="utf-8") as f:
            f.write(answer_text)

        # --- 阶段二：生成净化后的手写效果图 ---
        print(f"步骤 2/2: 正在生成净化排版的手写图...")
        
        # 修改后的 Prompt：强调删除无关内容
        draw_prompt = (
            f"根据提供的原始图片和以下解答，生成一张精美的手写解答长图。\n\n"
            f"【排版指令】：\n"
            f"1. **题目净化**：在图片顶部生成一个‘题目打印区’。请只保留原始图片中的题目文本和必要的几何图形，"
            f"彻底删除原始图片中的背景、杂物、手指、草稿痕迹或任何无关的边框。使其看起来像从干净的卷面上剪下来的贴纸。\n"
            f"2. **手写解答**：在题目下方，使用美观、工整的中文手写体书写解答过程。要求抄一遍题目，"
            f"推导过程严谨，使用标准的数学符号，逻辑清晰。\n"
            f"3. **视觉风格**：背景为干净的白色纸张。手机竖屏显示友好，文字清晰可读。\n\n"
            f"【参考解答内容】：\n{answer_text}"
        )

        with Image.open(image_path) as raw_img:
            image_draw_response = client.models.generate_content(
                model=MODEL_IMAGE,
                contents=[draw_prompt, raw_img],
                config=types.GenerateContentConfig(
                http_options=types.HttpOptions(
                    timeout=300000 # 5 分钟超时，绘图通常较慢
                )
            )
            )

        # 保存图片逻辑
        image_saved = False
        for part in image_draw_response.parts:
            if part.inline_data is not None:
                generated_image = part.as_image()
                generated_image.save(final_image_name)
                print(f"✅ 最终手写图（已净化题目）已保存: {final_image_name}")
                image_saved = True
        
        if not image_saved:
            print("⚠️ 绘图模型未返回图片数据。")

    except Exception as e:
        print(f"❌ 流程中断: {e}")

if __name__ == "__main__":
    for i in range(14, 15):
        target = f'./lilin-{i:06d}.png'
        print(f"正在处理文件: {target}")
        if os.path.exists(target):
            process_pipeline(target)
        else:
            print(f"错误：找不到文件 {target}")