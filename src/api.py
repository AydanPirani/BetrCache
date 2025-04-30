from enum import Enum
from dataclasses import dataclass
import requests
from openai import OpenAI
from typing import Optional, Union, List, Dict
import base64

class Provider(Enum):
    OPENAI = "openai"
    OPENROUTER = "openrouter"


@dataclass
class GPTOptions:
    model: str
    provider: Provider
    api_key: str
    prefix: str = ""
    image_path: Optional[str] = None  # optional image input


@dataclass
class EmbeddingOptions:
    model: str
    provider: Provider
    api_key: str


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_gpt_response(prompt: str, opts: GPTOptions) -> str:
    """
    Sends `opts.prefix + prompt` (plus an image if provided) to the specified
    model/provider, using opts.api_key for authentication.
    """
    # Initialize client with the given API key
    client = OpenAI(api_key=opts.api_key)

    if opts.image_path:
        base64_image = encode_image(opts.image_path)

        completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": prompt },
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
            input=prompt
        )
        return response.output_text


def get_embedding(input: str, options: EmbeddingOptions) -> list[float]:
    if options.provider == Provider.OPENROUTER:
        raise NotImplementedError("OpenRouter embeddings not supported")
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {options.api_key}",
    }
    payload = {"model": options.model, "input": input}
    
    resp = requests.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    return data["data"][0]["embedding"]
