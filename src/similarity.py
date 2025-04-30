from typing import List
from src.config import Modality


def cosine_similarity(modality: Modality, a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must be same length")

    if modality == Modality.TEXT:
        return text_cosine_similarity(a, b)
    
    if modality == Modality.MULTIMODAL:
        return multimodal_cosine_similarity(a, b)
        

def text_cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return [dot / (na * nb), 100]

def multimodal_cosine_similarity(a: List[float], b: List[float]) -> float:
    shift = len(a) // 2
    text_dot = sum(x * y for x, y in zip(a[:shift], b[:shift]))
    image_dot = sum(x * y for x, y in zip(a[shift:], b[shift:]))

    text_na = sum(x * x for x in a[:shift]) ** 0.5
    text_nb = sum(y * y for y in b[:shift]) ** 0.5

    image_na = sum(x * x for x in a[shift:]) ** 0.5
    image_nb = sum(y * y for y in b[shift:]) ** 0.5

    text_cosine_similarity = text_dot / (text_na * text_nb)
    image_cosine_similarity = image_dot / (image_na * image_nb)

    return [text_cosine_similarity, image_cosine_similarity]