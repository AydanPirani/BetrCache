import argparse

from PIL import Image

from src.api import get_embedding, get_gpt_response
from src.config import *
from src.ann_index import HnswAnnIndex
from src.cache_client import RedisClient
from src.cache import CacheConfig, EmbeddingCache
from src.api import GPTOptions, EmbeddingOptions, Provider, get_embedding, get_gpt_response
from src.similarity import cosine_similarity
from src.utils import logger, setup_logging

def query(
    image: Image.Image,
    prompt: str,
    gpt_opts: GPTOptions,
    emb_opts: EmbeddingOptions,
    cache: EmbeddingCache,
    threshold: int = 5,
    sim_threshold: float = 0.8,
) -> str:
    
    if image is not None:
        logger.info(f"Querying for image, {prompt}")
    else:
        logger.info(f"Querying for {prompt}")
    
    emb = get_embedding(image=image, prompt=prompt, options=emb_opts)
    candidates = cache.semantic_search("text", emb, threshold)

    if not candidates:
        logger.debug("No match found, querying LLM")
        resp = get_gpt_response(prompt=prompt, options=gpt_opts)
        cache.store_embedding("text", prompt, emb, resp)
        return resp

    # best = max(candidates, key=lambda d: cosine_similarity(d.embedding, emb))
    # score = cosine_similarity(best.embedding, emb)
    # logger.debug(f"Best match ({score}): {best.query} ({best.response})")
    # if score > sim_threshold:
    #     return best.response
    
    logger.debug("No good match found, querying LLM")
    resp = get_gpt_response(prompt, gpt_opts)
    logger.info("Got response from LLM")
    cache.store_embedding("text", prompt, emb, resp)
    return resp


def repl():
    text_ann_index = HnswAnnIndex(1000, TEXT_EMBEDDING_DIMENSION)
    text_client = RedisClient(REDIS_URL)
    text_client.delete("embeddings:text")

    image_ann_index = HnswAnnIndex(1000, IMAGE_EMBEDDING_DIMENSION)
    image_client = RedisClient(REDIS_URL)
    image_client.delete("embeddings:image")

    configs = {
        "text": CacheConfig(
            client=text_client,
            ann_index=text_ann_index,
            embedding_size=TEXT_EMBEDDING_DIMENSION
        ),
        "image": CacheConfig(
            client=image_client,
            ann_index=image_ann_index,
            embedding_size=IMAGE_EMBEDDING_DIMENSION
        ),
    }
    
    cache = EmbeddingCache(
        configs=configs,
        redis_key_prefix="embeddings",
        cache_ttl=3600,
    )

    while True:
        prompt = input("> ")
        resp = query(image=None, prompt=prompt, gpt_opts=gpt_opts, emb_opts=emb_opts, cache=cache, threshold=THRESHOLD, sim_threshold=SIMILARITY_THRESHOLD)
        # print(resp)
  
        # image_path = input("Enter the path to the image: ")
        # if image_path != None:
        #     try:
        #         image = Image.open(image_path)
        #     except Exception as e:
        #         print(f"Error loading image: {e}")
        #         continue
        
        # image = None
        # Prompt for text input
        # text_input = input("Enter the text query: ")

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
