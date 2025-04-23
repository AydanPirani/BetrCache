import argparse
import os
from dotenv import load_dotenv

from src.ann_index import HnswAnnIndex
from src.cache_client import RedisClient
from src.cache import Cache
from src.api import GPTOptions, EmbeddingOptions, Provider, get_embedding, get_gpt_response
from src.similarity import cosine_similarity
from src.utils import logger, setup_logging

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL")
DRAGONFLY_URL = os.getenv("DRAGONFLY_URL")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
THRESHOLD = int(os.getenv("THRESHOLD", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.8"))

LLM_PROVIDER = Provider.OPENROUTER
EMB_PROVIDER = Provider.OPENAI

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

def query(
    prompt: str,
    gpt_opts: GPTOptions,
    emb_opts: EmbeddingOptions,
    cache: Cache,
    threshold: int = 5,
    sim_threshold: float = 0.8,
) -> str:
    print("INIT", cache.ann_index.get_max_elements())
    logger.info(f"Querying for {prompt}")
    emb = get_embedding(prompt, emb_opts)
    candidates = cache.semantic_search(emb, threshold)

    if not candidates:
        logger.debug("No match found, querying LLM")
        resp = get_gpt_response(prompt, gpt_opts)
        cache.store_embedding(prompt, emb, resp)
        return resp

    best = max(candidates, key=lambda d: cosine_similarity(d.embedding, emb))
    score = cosine_similarity(best.embedding, emb)
    logger.debug(f"Best match ({score}): {best.query} ({best.response})")
    if score > sim_threshold:
        return best.response
    
    logger.debug("No good match found, querying LLM")
    resp = get_gpt_response(prompt, gpt_opts)
    logger.info("Got response from LLM")
    cache.store_embedding(prompt, emb, resp)
    print("AFTER", cache.ann_index.get_max_elements())
    return resp


def repl():
    index = HnswAnnIndex(1000, EMBEDDING_DIMENSION)
    client = RedisClient(DRAGONFLY_URL)
    client.delete("embeddings")
    cache = Cache(client, index, "embeddings", EMBEDDING_DIMENSION, cache_ttl=0)

    while True:
        prompt = input("> ")
        resp = query(prompt, gpt_opts, emb_opts, cache, THRESHOLD, SIMILARITY_THRESHOLD)
        print(resp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo application with stdout logging"
    )
        
    parser.add_argument(
        "--log-level", "--log", "-L",
        default="INFO",
        help="logging level; one of DEBUG, INFO, WARNING, ERROR, CRITICAL"
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger.debug("Debugging is now enabled")

    repl()
