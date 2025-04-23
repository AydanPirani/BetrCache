import json
from typing import List
from .cache_client import CacheClient
from .ann_index import ANNIndex
from .custom_types import EmbeddingData
from .utils import get_unix_seconds


class Cache:
    def __init__(
        self,
        client: CacheClient,
        ann_index: ANNIndex,
        redis_key: str,
        embedding_size: int,
        cache_ttl: int,
    ):
        self.client = client
        self.ann_index = ann_index
        self.embedding_size = embedding_size
        self.redis_key = redis_key
        self.cache_ttl = cache_ttl
        self.index_initialized = False
        self.current_id = 0

    def load_index(self) -> None:
        data = self.get_all_embeddings()
        if not data:
            self.ann_index.init_index(1, self.embedding_size)
            self.index_initialized = True
            return

        self.ann_index.init_index(len(data), self.embedding_size)
        max_id = 0
        for d in data:
            self.ann_index.add_pt(d.embedding, d.id)
            max_id = max(max_id, d.id)
        self.current_id = max_id + 1
        self.index_initialized = True

    def store_embedding(self, query: str, embedding: list[float], response: str) -> None:
        if len(embedding) != self.embedding_size:
            raise ValueError("Embedding size mismatch")
        eid = self.current_id
        self.current_id += 1
        data = EmbeddingData(
            id=eid, query=query, embedding=embedding, response=response, timestamp=get_unix_seconds()
        )
        self.client.h_set(self.redis_key, str(eid), data.json())
        if self.cache_ttl:
            self.client.expire(self.redis_key, self.cache_ttl)

        if not self.index_initialized:
            self.ann_index.init_index(1000, self.embedding_size)
            self.index_initialized = True

        if self.ann_index.get_curr_ct() > self.ann_index.get_max_elements():
            self.ann_index.resize(self.ann_index.get_curr_ct() + 1000)

        self.ann_index.add_pt(embedding, eid)

    def semantic_search(self, embedding: list[float], k: int) -> List[EmbeddingData]:
        if len(embedding) != self.embedding_size:
            raise ValueError("Embedding size mismatch")
        if not self.index_initialized:
            self.load_index()

        curr = self.ann_index.get_curr_ct()
        k = min(k, curr)
        if k == 0:
            return []

        results = self.ann_index.search_knn(embedding, k)
        ids = [str(r[0]) for r in results]
        raw = self.client.hm_get(self.redis_key, ids)
        out: list[EmbeddingData] = []
        for item in raw:
            if item:
                out.append(EmbeddingData.parse_raw(item))
        return out

    def get_all_embeddings(self) -> List[EmbeddingData]:
        raw = self.client.h_get_all(self.redis_key)
        return [EmbeddingData.parse_raw(v) for v in raw.values()]
