from PIL import Image
import requests
import torch
from transformers import CLIPProcessor, CLIPModel

# 加载 TinyCLIP 模型
model_path = "../TinyCLIP-ViT-61M-32-Text-29M-LAION400M"

model = CLIPModel.from_pretrained(model_path)
processor = CLIPProcessor.from_pretrained(model_path)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

texts = ["a photo of a cat", "a photo of a dog"]

# ========== 提取文本特征 ==========
text_inputs = processor(text=texts, return_tensors="pt", padding=True)
text_features = model.get_text_features(**text_inputs)   # [batch_size, hidden_dim]
text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # 归一化

# ========== 提取图片特征 ==========
image_inputs = processor(images=image, return_tensors="pt")
image_features = model.get_image_features(**image_inputs)  # [1, hidden_dim]
image_features = image_features / image_features.norm(dim=-1, keepdim=True)

# ========== 计算相似度 ==========
similarity = image_features @ text_features.T  # [1, num_texts]
probs = similarity.softmax(dim=-1)

print("相似度:", similarity)
print("概率:", probs)
