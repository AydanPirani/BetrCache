from enum import Enum
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

class Provider(Enum):
    OPENAI = "openai"
    OPENROUTER = "openrouter"

class Modality(Enum):
    TEXT = "text"
    IMAGE = "image"
    MULTIMODAL = "multimodal"


@dataclass
class GPTOptions:
    model: str
    provider: Provider
    api_key: str
    prefix: str


@dataclass
class EmbeddingOptions:
    model: str
    provider: Provider
    api_key: str


LLM_PROVIDER = Provider.OPENAI
EMB_PROVIDER = Provider.OPENAI

REDIS_URL = os.getenv("REDIS_URL")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

TEXT_EMBEDDING_DIMENSION = int(os.getenv("TEXT_EMBEDDING_DIMENSION", "1536"))
IMAGE_EMBEDDING_DIMENSION = int(os.getenv("IMAGE_EMBEDDING_DIMENSION", "512"))
MULTIMODAL_EMBEDDING_DIMENSION = int(os.getenv("MULTIMODAL_EMBEDDING_DIMENSION", "768"))

THRESHOLD = int(os.getenv("THRESHOLD", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

gpt_key = OPENAI_KEY if LLM_PROVIDER == Provider.OPENAI else OPENROUTER_KEY
emb_key = OPENAI_KEY if EMB_PROVIDER == Provider.OPENAI else OPENROUTER_KEY

gpt_opts = GPTOptions(
    model=LLM_MODEL,
    provider=LLM_PROVIDER,
    api_key=gpt_key,
    prefix="You are a search assistant. Give me a response in 5 sentences."
)

emb_opts = EmbeddingOptions(
    model=EMBEDDINGS_MODEL,
    provider=EMB_PROVIDER,
    api_key=emb_key
)
