from src.config import *
from src.ann_index import HnswAnnIndex
from src.cache_client import RedisClient
from src.cache import CacheConfig, EmbeddingCache
from src.api import LLMInput, get_gpt_response
from src.judge import SimilarityScorer
from main import query
import time

from src.dataset import get_dataset

def evaluate_flickr30k():
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

    dataset = get_dataset()
    print("got dataset")

    file = open("judge.csv", "w")

    for img in dataset:
        captions = dataset[img]
        if len(captions) < 2:
            continue

        caption1 = captions[0]
        caption2 = captions[1]
        question = "Is this an appropriate caption for this image: "

        image_path = "flickr30k-images/" + img
        
        
        first_llm_input = LLMInput(text=question + caption1, image=image_path)
        second_llm_input = LLMInput(text=question + caption2, image=image_path)

        first_output = query(llm_input=first_llm_input, gpt_opts=gpt_opts, emb_opts=emb_opts, cache=cache, threshold=THRESHOLD, sim_threshold=SIMILARITY_THRESHOLD)

        act_output = query(llm_input=second_llm_input, gpt_opts=gpt_opts, emb_opts=emb_opts, cache=cache, threshold=THRESHOLD, sim_threshold=SIMILARITY_THRESHOLD)

        exp_output = get_gpt_response(llm_input=second_llm_input, options=gpt_opts)


        scorer = SimilarityScorer(gpt_opts)
        score, _ = scorer.similarity_score(act_output.text, exp_output)

        print(score)

        hit_first_input = act_output.is_hit and (first_llm_input.text == act_output.best_candidate.query)
        file.write(f"{img},{caption1},{caption2},{act_output.is_hit},{hit_first_input},{score}\n")


evaluate_flickr30k()
