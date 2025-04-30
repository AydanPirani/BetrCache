import argparse
import os
from dotenv import load_dotenv
from PIL import Image
# from src.api import GPTOptions, EmbeddingOptions, Provider, get_embedding, get_gpt_response
from transformers import AutoTokenizer, CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

image1 = Image.open("dawg1.jpeg")
image2 = Image.open("dawg2.jpeg")
text_input = "describe image"

print("TEXT ENCODING!!")
if text_input is not None:
    inputs = tokenizer([text_input], padding=True, return_tensors="pt")
    text_emb = model.get_text_features(**inputs)
print("text: ", text_emb)

print("IMAGE ENCODING!!")

inputs1 = processor(images=image1, return_tensors="pt")
img_emb1 = model.get_image_features(**inputs1)

inputs2 = processor(images=image2, return_tensors="pt")
img_emb2 = model.get_image_features(**inputs2)

img_emb1 = img_emb1 / img_emb1.norm(p=2, dim=-1, keepdim=True)
img_emb2 = img_emb2 / img_emb2.norm(p=2, dim=-1, keepdim=True)

# Compute cosine similarity between the two image embeddings
cosine_sim = cosine_similarity(img_emb1.detach().numpy(), img_emb2.detach().numpy())
cosine_sim

print(cosine_sim)