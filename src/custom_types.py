from pydantic import BaseModel
from typing import List


class EmbeddingData(BaseModel):
    id: int
    query: str
    image: str = ""
    embedding: List[float]
    response: str
    timestamp: int
