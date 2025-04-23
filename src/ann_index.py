from abc import ABC, abstractmethod
import hnswlib
from typing import List, Tuple


class ANNIndex(ABC):
    @abstractmethod
    def init_index(self, max_elements: int, dimension: int) -> None:
        ...

    @abstractmethod
    def add_pt(self, point: List[float], id: int) -> None:
        ...

    @abstractmethod
    def get_curr_ct(self) -> int:
        ...

    @abstractmethod
    def get_max_elements(self) -> int:
        ...

    @abstractmethod
    def resize(self, new_size: int) -> None:
        ...

    @abstractmethod
    def search_knn(self, query: List[float], k: int) -> List[Tuple[int, float]]:
        ...


class HnswAnnIndex(ANNIndex):
    def __init__(self, max_elements: int, dimension: int):
        self.dimension = dimension
        self.max_elements = max_elements
        self.points: List[Tuple[List[float], int]] = []
        self._init_hnsw()

    def _init_hnsw(self):
        self.index = hnswlib.Index(space='cosine', dim=self.dimension)
        # M=16, ef_construction=100 are defaults you can tune
        self.index.init_index(max_elements=self.max_elements, ef_construction=100, M=16)

    def init_index(self, max_elements: int, dimension: int) -> None:
        self.dimension = dimension
        self.max_elements = max_elements
        self.points.clear()
        self._init_hnsw()

    def add_pt(self, point: List[float], id: int) -> None:
        if len(point) != self.dimension:
            raise ValueError("Point dimensions don't match!")
        self.index.add_items(point, ids=[id])
        self.points.append((point, id))

    def get_curr_ct(self) -> int:
        return self.index.get_current_count()

    def get_max_elements(self) -> int:
        return self.max_elements

    def resize(self, new_size: int) -> None:
        # Rebuild a larger index
        old_points = list(self.points)
        self.max_elements = new_size
        self._init_hnsw()
        for pt, pid in old_points:
            self.index.add_items(pt, ids=[pid])
        self.points = old_points

    def search_knn(self, query: List[float], k: int) -> List[Tuple[int, float]]:
        labels, distances = self.index.knn_query(query, k=k)
        # hnswlib returns lists of arrays
        return list(zip(labels[0].tolist(), distances[0].tolist()))
