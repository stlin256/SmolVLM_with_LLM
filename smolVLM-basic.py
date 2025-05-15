'''
a basic python script to run smolVLM
基础的使用smol-VLM进行图像推理的脚本
'''

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

DEVICE = "cuda" #in order to use flash attention

# model_name = "HuggingFaceTB/SmolVLM-256M-Instruct" #load from huggingface
model_name = "./SmolVLM-256M-Instruct" #load from local file
print(f"Loading model from {model_name}...")
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2",
    device_map="cuda"  # force on GPU
).to(DEVICE)

# Create messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Can you describe this image?"}
        ]
    },
]

# Prepare inputs
def chat(image):
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = inputs.to(DEVICE)

    # Generate outputs
    generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    print(generated_texts[0])

def main():
    img_add=r""
    while img_add!="exit":
        image_add = input("input the url of image: ")
        print(f"loading image: {image_add}")
        image = load_image(image_add)
        print("generating text...")
        chat(image)

if __name__ == "__main__":
    main()