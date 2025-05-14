from openai import OpenAI
import requests
from PIL import Image
from transformers import AutoTokenizer, CLIPProcessor, CLIPModel
from src.config import *
from src.utils import logger
from src.custom_types import EmbeddingData
from typing import Optional
import base64
from dataclasses import dataclass

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

@dataclass
class LLMInput:
    text: str
    image: str = ""

@dataclass
class LLMOutput:
    text: str = ""
    is_hit: bool = False
    best_candidate: EmbeddingData = None


def get_gpt_response(llm_input: LLMInput, options: GPTOptions) -> str:
    logger.debug("IN GPT RESPONSE")
    client = OpenAI(api_key=options.api_key)

    if llm_input.image != "":
        base64_image = encode_image(llm_input.image)

        completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": options.prefix + llm_input.text },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        )
        return completion.choices[0].message.content
    else:
        response = client.responses.create(
            model="gpt-4.1",
            input=options.prefix + llm_input.text
        )
    return response.output_text


def get_embedding(llm_input: LLMInput, options: EmbeddingOptions) -> list[float]:
    print(llm_input.image)
    if len(llm_input.image) == 0: 
        url = "https://api.openai.com/v1/embeddings"

        prompt = llm_input.text

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {options.api_key}",
        }
        payload = {"model": options.model, "input": prompt}
        
        resp = requests.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return [data["data"][0]["embedding"], []]
    else: 
        #image
        image_path = llm_input.image
        img = Image.open(image_path)
        inputs = processor(images=img, return_tensors="pt")
        img_emb = model.get_image_features(**inputs)
        img_emb = img_emb.detach().numpy().flatten()

        #text
        inputs = tokenizer([llm_input.text], padding=True, return_tensors="pt")
        text_emb = model.get_text_features(**inputs)
        text_emb = text_emb.detach().numpy().flatten()
        
        return [text_emb, img_emb]

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
