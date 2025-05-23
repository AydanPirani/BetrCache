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

    scorer = SimilarityScorer(gpt_opts, emb_opts)

    dataset = get_dataset(num_return=50)
    print("got dataset")

    file = open("embedding.tsv", "w")
    i = 0
    for img in dataset:
        print(i)
        i += 1

        captions = dataset[img]
        if len(captions) < 2:
            continue

        #Creating Similar Inputs
        caption1 = captions[0]
        caption2 = captions[1]
        question = "Is this an appropriate caption for this image: "

        image_path = "flickr30k-images/" + img
        
        
        first_llm_input = LLMInput(text=question + caption1, image=image_path)
        second_llm_input = LLMInput(text=question + caption2, image=image_path)

        #Queries
        first_output = query(llm_input=first_llm_input, gpt_opts=gpt_opts, emb_opts=emb_opts, cache=cache, threshold=THRESHOLD, sim_threshold=SIMILARITY_THRESHOLD)

        start_time = time.time()
        act_output = query(llm_input=second_llm_input, gpt_opts=gpt_opts, emb_opts=emb_opts, cache=cache, threshold=THRESHOLD, sim_threshold=SIMILARITY_THRESHOLD)
        latency = time.time() - start_time

        exp_output = get_gpt_response(llm_input=second_llm_input, options=gpt_opts)

        #Metrics
        cache_llm_score = scorer.similarity_score(first_output.text, exp_output)
        cache_emb_score = scorer.embeddings_similarity(first_output.text, exp_output)

        if first_output.text != act_output.text:
            true_llm_score = scorer.similarity_score(act_output.text, exp_output)
            true_emb_score = scorer.embeddings_similarity(act_output.text, exp_output)
        else:
            true_llm_score = cache_llm_score
            true_emb_score = cache_emb_score

        hit_first_input = act_output.is_hit and (first_llm_input.text == act_output.best_candidate.query)
        file.write(f"{img}\t{caption1}\t{caption2}\t{act_output.is_hit}\t{hit_first_input}\t{cache_llm_score}\t{cache_emb_score}\t{true_llm_score}\t{true_emb_score}\t{latency}\n")
        file.flush()

if __name__ == "__main__":
    evaluate_flickr30k()
