import requests
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from src.config import *
from src.utils import logger


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def get_gpt_response(prompt: str, options: GPTOptions) -> str:
    print("IN GPT RESPONSE")
    url = (
        "https://api.openai.com/v1/responses"
        if options.provider == Provider.OPENAI
        else "https://openrouter.ai/api/v1/chat/completions"
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {options.api_key}",
    }

    payload = {
        "model": options.model,
        "messages": [{"role": "user", "content": options.prefix + prompt}],
    }
    
    logger.info(f"{url}, {payload}, {headers}")
    
    resp = requests.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def get_embedding(image: Image.Image, prompt: str, options: EmbeddingOptions) -> list[float]:
    if options.provider == Provider.OPENROUTER:
        raise NotImplementedError("OpenRouter embeddings not supported")
    
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {options.api_key}",
    }
    payload = {"model": options.model, "input": prompt}
    
    resp = requests.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    return data["data"][0]["embedding"]
    # if img is not None: 
        

    # if isinstance(input, str): # text
    #     inputs = processor(text=[input], return_tensors="pt", padding=True)
    #     text_features = model.get_text_features(**inputs)
    #     return text_features.detach().numpy().flatten()
    # elif isinstance(input, Image): # images
    #     inputs = processor(images=input, return_tensors="pt")
    #     image_features = model.get_image_features(**inputs)
    #     return image_features.detach().numpy().flatten()
    # else:
    #     raise ValueError("Unsupported input type")
