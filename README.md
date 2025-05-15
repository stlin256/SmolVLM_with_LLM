
# SmolVLM&LLM

### Combining SmolVLM, SmolVLM2-video models with LLM (Qwen3)
<a href="#1">中文介绍</a>

----

### Why combine VLM and LLM?

Small-parameter visual models *are suitable for edge deployment, but due to their parameter limitations, capabilities in logical reasoning and multilingual support are **limited**. By combining them with a remote LLM, we can **enhance overall performance**.

Using VLM on the edge and sending its output to a cloud-based LLM service for further reasoning helps **reduce network overhead**.

----

### Repository Contents

This repository provides four basic Python scripts that integrate the [SmolVLM-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct),  
[SmolVLM2-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct) models with LLMs, compensating for their limitations in multilingual and logical reasoning capabilities due to smaller parameter counts, while offering new insights into applying small-parameter VLM models.

- `smolVLM-LLM-basic.py` provides a basic usage example of SmolVLM-256M-Instruct. You can load an image via URL for inference.

- `SmolVLM-LLM.py` demonstrates how to combine SmolVLM-256M-Instruct with a remote LLM service (Qwen3-0.6B) to achieve **image content reasoning** and **efficient translation**.

- `SmolVLM-LLM-int4.py` shows how to use SmolVLM-256M-Instruct (int4 quantized version) together with a remote LLM service (Qwen3-0.6B) for similar tasks: **image content reasoning** and **efficient translation**.

- `SmolVLM2-video-LLM.py` demonstrates how to combine SmolVLM2-256M-Video-Instruct with a remote LLM service (Qwen3-0.6B) to perform **video content reasoning**, **output cleaning**, **content summarization**, and **efficient translation**.

----

### Models Used

[SmolVLM-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct) and  
[SmolVLM2-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct)  
are lightweight and efficient vision models developed by HuggingFaceTB.

[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) is the 0.6B parameter version of Alibaba's Qwen3 model, known for strong translation capabilities and fast response speed.

----
<div id="1"></div>

# SmolVLM&LLM

### 将SmolVLM、SmolVLM2-video模型和LLM(Qwen3)结合使用

----
### 为什么将VLM和LLM结合起来用？
小参数量的视觉模型*适合在端侧部署，但是由于其参数量的局限性，逻辑推理能力和多语言能力**受限**，将其**与远程的LLM结合使用**，则能**提升使用效果**。

端侧使用VLM，再将输出内容送到云端LLM服务进行更进一步的推理，可以**减少网络开销**。

----
### 仓库内容
本仓库提供四个基础的python脚本，将[SmolVLM-Istruct](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)、
[SmolVLM2-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct)模型与LLM结合使用，弥补二者由于参数量原因导致的多语言和逻辑推理能力上的不足，同时为小参数量VLM模型的应用提供新的思路。

`smolVLM-LLM-basic.py`中提供了基础的SmolVLM-256M-Istruct的使用样例，输入图片url加载图像即可进行推理。

`SmolVLM-LLM.py`中提供了将SmolVLM-256M-Istruct和远程LLM服务（Qwen3-0.6b）结合使用，实现**图片内容推理**、**高效翻译**的使用样例。

`SmolVLM-LLM-int4.py`中提供了将SmolVLM-256M-Istruct（int4量化）和远程LLM服务（Qwen3-0.6b）结合使用，实现**图片内容推理**、**高效翻译**的使用样例。

`SmolVLM2-video-LLM.py`中提供了将SmolVLM2-256M-Video-Instruct和远程LLM服务（Qwen3-0.6b）结合使用，实现**视频内容推理**、**输出内容清洗**、**内容总结**和**高效翻译**的使用样例。

----
### 使用模型
[SmolVLM-Istruct](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)和
[SmolVLM2-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct
)是由HuggingFaceTB开发的视觉模型，轻量且好用。

[Qwen3-0.6b](https://huggingface.co/Qwen/Qwen3-0.6B)是由阿里开发的Qwen3模型的0.6B参数版本，翻译能力强，速度快。
