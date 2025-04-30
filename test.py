import argparse
import os
from dotenv import load_dotenv
from PIL import Image
from src.api import GPTOptions, EmbeddingOptions, Provider, get_embedding, get_gpt_response
from transformers import AutoTokenizer, CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("gekko.jpg")
text_input = "describe image"

print("TEXT ENCODING!!")
if text_input is not None:
    inputs = tokenizer([text_input], padding=True, return_tensors="pt")
    text_emb = model.get_text_features(**inputs)
print("text: ", text_emb)

print("IMAGE ENCODING!!")
if image is not None: 
    inputs = processor(images=image, return_tensors="pt")
    img_emb = model.get_image_features(**inputs)
    # img_emb = image_features.detach().numpy().flatten()
print("img: ", img_emb)