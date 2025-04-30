from typing import List
from .cache_client import CacheClient
from .ann_index import ANNIndex
from .custom_types import EmbeddingData
from .utils import get_unix_seconds, logger

from dataclasses import dataclass
from typing import Dict, List
from .cache_client import CacheClient
from .ann_index import ANNIndex
from .custom_types import EmbeddingData
from .utils import get_unix_seconds, logger

@dataclass
class CacheConfig:
    client: CacheClient
    ann_index: ANNIndex
    embedding_size: int
    current_id: int = 0
    index_initialized: bool = False
    initial_size: int = 1000

class EmbeddingCache:
    def __init__(
        self,
        configs: Dict[str, CacheConfig],
        redis_key_prefix: str,
        cache_ttl: int,
    ):
        """
        configs: map from modality name (e.g. "text", "image") to its CacheConfig
        redis_key_prefix: base key (we’ll append the modality)
        cache_ttl: seconds until each hash key expires
        """
        self.configs = configs
        self.redis_key_prefix = redis_key_prefix
        self.cache_ttl = cache_ttl


    def _redis_key(self, modality: str) -> str:
        return f"{self.redis_key_prefix}:{modality}"


    def _load_index(self, modality: str) -> None:
        cfg = self.configs[modality]
        logger.debug(f"Loading '{modality}' index")
        data = self.get_all_embeddings(modality)
        if not data and not cfg.index_initialized:
            cfg.ann_index.init_index(cfg.initial_size, cfg.embedding_size)
        else:
            cfg.ann_index.init_index(len(data), cfg.embedding_size)
            max_id = 0
            for d in data:
                cfg.ann_index.add_pt(d.embedding, d.id)
                max_id = max(max_id, d.id)
            cfg.current_id = max_id + 1
        cfg.index_initialized = True


    def store_embedding(
        self,
        modality: str,
        query: str,
        embedding: List[float],
        response: str,
    ) -> None:
        cfg = self.configs[modality]
        if len(embedding) != cfg.embedding_size:
            logger.error(f"Size mismatch for {modality}: {len(embedding)} vs {cfg.embedding_size}")
            raise ValueError("Embedding size mismatch")

        eid = cfg.current_id
        cfg.current_id += 1

        payload = EmbeddingData(
            id=eid,
            query=query,
            embedding=embedding,
            response=response,
            timestamp=get_unix_seconds()
        )

        payload_json = payload.model_dump_json()

        key = self._redis_key(modality)
        cfg.client.h_set(key, str(eid), payload_json)
        if self.cache_ttl:
            cfg.client.expire(key, self.cache_ttl)

        if not cfg.index_initialized:
            cfg.ann_index.init_index(cfg.initial_size, cfg.embedding_size)
            cfg.index_initialized = True

        if cfg.ann_index.get_curr_ct() >= cfg.ann_index.get_max_elements():
            new_size = cfg.ann_index.get_curr_ct() + 1000
            logger.debug(f"Resizing '{modality}' ANN index → {new_size}")
            cfg.ann_index.resize(new_size)

        cfg.ann_index.add_pt(embedding, eid)

    def semantic_search(
        self,
        modality: str,
        embedding: List[float],
        k: int
    ) -> List[EmbeddingData]:
        cfg = self.configs[modality]
        if len(embedding) != cfg.embedding_size:
            logger.error(f"Size mismatch for {modality}: {len(embedding)} vs {cfg.embedding_size}")
            raise ValueError("Embedding size mismatch")

        if not cfg.index_initialized:
            self._load_index(modality)

        curr = cfg.ann_index.get_curr_ct()
        if curr == 0:
            return []

        k = min(k, curr)
        results = cfg.ann_index.search_knn(embedding, k)
        ids = [str(r[0]) for r in results]
        raw = cfg.client.hm_get(self._redis_key(modality), ids)

        return [
            EmbeddingData.parse_raw(item)
            for item in raw
            if item is not None
        ]


    def get_all_embeddings(self, modality: str) -> List[EmbeddingData]:
        raw = self.configs[modality].client.h_get_all(self._redis_key(modality))
        return [EmbeddingData.parse_raw(v) for v in raw.values()]
