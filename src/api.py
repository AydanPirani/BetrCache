from openai import OpenAI
import requests
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from src.config import *
from src.utils import logger
from typing import Optional
import base64
from dataclasses import dataclass

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

@dataclass
class LLMInput:
    text: str
    image: Optional[str] = None


def get_gpt_response(llm_input: LLMInput, options: GPTOptions) -> str:
    logger.debug("IN GPT RESPONSE")
    client = OpenAI(api_key=options.api_key)

    if llm_input.image is not None:
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
    if options.provider == Provider.OPENROUTER:
        raise NotImplementedError("OpenRouter embeddings not supported")

    url = "https://api.openai.com/v1/embeddings" if options.provider == Provider.OPENAI else "https://openrouter.ai/api/v1/embeddings"

    prompt = llm_input.text

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

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# def get_gpt_response(prompt: str, options: GPTOptions) -> str:
#     """
#     Sends `opts.prefix + prompt` (plus an image if provided) to the specified
#     model/provider, using opts.api_key for authentication.
#     """
#     # Initialize client with the given API key
#     client = OpenAI(api_key=opts.api_key)

#     if opts.image_path:
#         base64_image = encode_image(opts.image_path)

#         completion = client.chat.completions.create(
#         model="gpt-4.1",
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     { "type": "text", "text": prompt },
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": f"data:image/jpeg;base64,{base64_image}",
#                         },
#                     },
#                 ],
#             }
#         ],
#         )
#         return completion.choices[0].message.content
#     else:
#         response = client.responses.create(
#             model="gpt-4.1",
#             input=prompt
#         )
#         return response.output_text
