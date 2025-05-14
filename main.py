import argparse
from time import sleep
from typing import Optional

from PIL import Image

from src.api import get_embedding, get_gpt_response, LLMInput, LLMOutput
from src.config import *
from src.ann_index import HnswAnnIndex
from src.cache_client import RedisClient
from src.cache import CacheConfig, EmbeddingCache
from src.api import GPTOptions, EmbeddingOptions, Provider, get_embedding, get_gpt_response
from src.similarity import cosine_similarity
from src.utils import logger, setup_logging
import time

def query(
    llm_input: LLMInput,
    gpt_opts: GPTOptions,
    emb_opts: EmbeddingOptions,
    cache: EmbeddingCache,
    threshold: int = 5,
    sim_threshold: float = 0.8,
) -> LLMOutput:
    
    prompt = llm_input.text
    output = LLMOutput()

    modality = Modality.MULTIMODAL if llm_input.image != "" else Modality.TEXT
    
    if llm_input.image != "" :
        logger.info(f"Querying for image, {prompt}")
    else:
        logger.info(f"Querying for {prompt}")

    text_emb, img_emb = get_embedding(llm_input=llm_input, options=emb_opts)
    emb = [*text_emb, *img_emb]

    candidates = cache.semantic_search(modality=modality, embedding=emb, k=threshold)
    

    def red(c):
        x, y = cosine_similarity(modality, emb, c.embedding)
        return x**2 + y**2

    def cmp(c):
        text_score, image_score = cosine_similarity(modality, a=emb, b=c.embedding)
        return text_score > sim_threshold and image_score > sim_threshold

    candidates = list(filter(cmp, candidates))
    
    if not candidates:
        logger.debug("No match found, querying LLM")
        resp = get_gpt_response(llm_input=llm_input, options=gpt_opts)
        cache.store_embedding(modality=modality, llm_input=llm_input, embedding=emb, response=resp)

        output.text = resp
        return output
    
    best = max(candidates, key=red)
    text_score, image_score = cosine_similarity(modality, a=emb, b=best.embedding)
    
    logger.debug(f"Best match ({text_score}, {image_score}): {best.query} ({best.response})")
    if text_score > sim_threshold and image_score > sim_threshold:
        output.text = best.response
        output.is_hit = True
        output.best_candidate = best

        return output
    
    logger.debug("No good match found, querying LLM")
    resp = get_gpt_response(llm_input=llm_input, options=gpt_opts)
    logger.info("Got response from LLM")
    cache.store_embedding(modality=modality, llm_input=llm_input, embedding=emb, response=resp)

    output.text = resp
    return output


def repl():

    text_ann_index = HnswAnnIndex(1000, TEXT_EMBEDDING_DIMENSION)
    text_client = RedisClient(REDIS_URL)
    text_client.delete("embeddings:text")

    multimodal_ann_index = HnswAnnIndex(1000, MULTIMODAL_EMBEDDING_DIMENSION)
    multimodal_client = RedisClient(REDIS_URL)
    multimodal_client.delete("embeddings:multimodal")

    configs = {
        Modality.TEXT: CacheConfig(
            client=text_client,
            ann_index=text_ann_index,
            embedding_size=TEXT_EMBEDDING_DIMENSION
        ),
        Modality.MULTIMODAL: CacheConfig(
            client=multimodal_client,
            ann_index=multimodal_ann_index,
            embedding_size=MULTIMODAL_EMBEDDING_DIMENSION
        )
    }
    
    cache = EmbeddingCache(
        configs=configs,
        redis_key_prefix="embeddings",
        cache_ttl=3600,
    )

    while True:
        text = input("Enter text prompt: ")
        image_path = input("Enter image path: ")

        llm_input = LLMInput(text=text, image=image_path)
        
        start_time = time.time()
        exp_resp = query(llm_input=llm_input, gpt_opts=gpt_opts, emb_opts=emb_opts, cache=cache, threshold=THRESHOLD, sim_threshold=SIMILARITY_THRESHOLD)
        act_resp = get_gpt_response(llm_input=llm_input, options=gpt_opts)
        latency = time.time() - start_time

        print(f"Response: {resp}")
        print(f"Query latency: {latency:.3f} seconds")
        print(resp)
        sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo application with stdout logging"
    )
        
    parser.add_argument(
        "--log-level", "--log", "-L",
        default="DEBUG",
        help="logging level; one of DEBUG, INFO, WARNING, ERROR, CRITICAL"
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger.debug("Debugging is now enabled")

    repl()
