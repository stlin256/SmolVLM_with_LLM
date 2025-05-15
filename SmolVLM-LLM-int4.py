'''
a little python script to connect SmolVLM with llm(int4 quantization)
一个将SmolVLM和LLM配合使用的脚本(使用int4量化)
由于SmolVLM的多语言支持不是很到位，特别是256M的小参数量版本，输出的中文简直是脸滚键盘，使用外置的大模型进行翻译是很有必要的

同时，可以通过提示词工程的方式，让SmolVLM描述内容，再使用别的大模型进行推理，起到一个手动“Moe”的作用
在侧端部署小参数量VLM，在云端部署大参数量LLM，可以减轻侧端性能需求，降低带宽占用，同时提升使用表现
在此推荐最新的Qwen3模型，其翻译能力很强大，0.6B参数量的模型即可满足翻译需求
'''

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from transformers.image_utils import load_image
from openai import OpenAI



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# check cuda device
if DEVICE == "cpu":
    raise RuntimeError("CUDA is all you need.")

# model_name = "HuggingFaceTB/SmolVLM-256M-Instruct" #load from huggingface
model_name = "./SmolVLM-256M-Instruct" #load from local files
model_name = "./SmolVLM-256M-Instruct"
print(f"从 {model_name} 加载模型...")

# set int 4 quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# loading
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16
).to(DEVICE)


# creating message
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Overview the picture"}
        ]
    },
]


def chat(image):

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = inputs.to(DEVICE)

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    # set OPENAI client; I use my local lmstudio service here
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    print(f"smolVLM输出：{generated_texts[0]}")
    # translate prompt
    translation_prompt = f"将英文文本翻译成中文,只翻译冒号后的输出内容：\n\n{generated_texts[0]}"

    # I recommend qwen3 for translating. because of its multi-language support.
    response = client.chat.completions.create(
        model="qwen3-0.6b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that translates English to Chinese./no_think"},
            {"role": "user", "content": translation_prompt}
        ],
        max_tokens=500
    )

    # get translation
    translation = response.choices[0].message.content.strip()
    print(f"QWEN3输出结果{translation}")



def main():

    while True:
        image_add = input("请输入图片URL（输入'exit'退出）：")
        if image_add.lower() == "exit":
            break
        print(f"正在加载图片：{image_add}")
        image = load_image(image_add)
        print("正在生成文本...")
        chat(image)


if __name__ == "__main__":
    main()