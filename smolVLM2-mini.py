'''
a little python script to connect SmolVLM2-video with llm
一个将SmolVLM2和LLM配合使用的脚本
由于SmolVLM2的多语言支持不是很到位，特别是256M的小参数量版本，输出的中文简直是脸滚键盘，使用外置的大模型进行翻译是很有必要的
SmolVLM2会输出大量错乱的标识，很有必要要求LLM忽视标识内容

同时，可以通过提示词工程的方式，让SmolVLM2描述内容，再使用别的大模型进行推理，起到一个手动“Moe”的作用
在侧端部署小参数量VLM，在云端部署大参数量LLM，可以减轻侧端性能需求，降低带宽占用，同时提升使用表现
在此推荐最新的Qwen3模型，其翻译能力很强大，0.6B参数量的模型即可满足翻译需求
'''
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from openai import OpenAI
from moviepy.video.io.VideoFileClip import VideoFileClip
model_path = "SmolVLM2-256M-Video-Instruct"


processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16
).to("cuda")

def get_duration_from_moviepy(url):
    clip = VideoFileClip(url)
    return clip.duration

def chat(video_path,video_length,token_ratio=64):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "path": video_path},
                {"type": "text", "text": "Describe this video"}
            ]
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    generated_ids = model.generate(**inputs, do_sample=True, max_new_tokens=token_ratio*video_length)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    print(generated_texts[0])

    # set OPENAI client; I use my local lmstudio service here
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    print(f"smolVLM2-video输出：{generated_texts[0]}")
    # translate prompt
    translation_prompt = f"请将以下英文文本翻译成中文,只翻译输出的内容，不包括标识：\n/no_think\n{generated_texts[0]}"#use your target language and emphasize not outputting labels

    # I recommend qwen3 for translating. because of its multi-language support.
    response = client.chat.completions.create(
        model="qwen3-0.6b",
        messages=[
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": translation_prompt}
        ],
        max_tokens=(token_ratio+8)*video_length
    )

    # get translation
    translation = response.choices[0].message.content.strip()
    print(f"QWEN3输出结果{translation}")

def main():
    while True:
        video_path = input("请输入视频url:")
        chat(video_path,int(get_duration_from_moviepy(video_path)))


if __name__ == "__main__":
    main()
