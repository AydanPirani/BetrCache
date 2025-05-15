from openai import OpenAI
from src.config import GPTOptions, EmbeddingOptions
from src.api import get_embedding, LLMInput
from math import sqrt

MAX_SCORE = 100

class SimilarityScorer:
    def __init__(self, gpt_options: GPTOptions, embedding_options: EmbeddingOptions):
        self.client = OpenAI(api_key=gpt_options.api_key)
        self.model = gpt_options.model
        self.embedding_options = embedding_options

    def similarity_score(self, text_a: str, text_b: str) -> float:
        """
        Returns a similarity score between 0 (no similarity) and 1000 (identical) for text_a and text_b.
        """
        system_message = {
            "role": "system",
            "content": (
                "You are a semantic similarity evaluator. "
                "Given two texts, compute how semantically similar they are on a scale from 0 to 1000, "
                "where 0 means completely unrelated and 100 means identical in meaning. "
                "Respond in the following format XX.YY;<REASON>, where <REASON> should be replaced with a short scoring rationale."
            )
        }
        user_message = {
            "role": "user",
            "content": (
                f"Text A:\n{text_a}\n\n"
                f"Text B:\n{text_b}\n\n"
                "Please provide the similarity score."
            )
        }

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[system_message, user_message],
            temperature=0,
            top_p=1,
        )
        content = completion.choices[0].message.content.strip()
        score, _ = content.split(";", 1)
        return float(score.strip())/MAX_SCORE
    
    def embeddings_similarity(self, text_a: str, text_b: str) -> float:
        # Compute cosine similarity
        a = LLMInput(text=text_a)
        b = LLMInput(text=text_b)
        emb_a = get_embedding(a, self.embedding_options)[0]
        emb_b = get_embedding(b, self.embedding_options)[0]

        dot = sum(x * y for x, y in zip(emb_a, emb_b))
        norm_a = sqrt(sum(x * x for x in emb_a))
        norm_b = sqrt(sum(y * y for y in emb_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    opts = GPTOptions(
        model=os.getenv("LLM_MODEL"),
        provider=None,  # not used here
        api_key=os.getenv("OPENAI_API_KEY"),
        prefix=""
    )

    scorer = SimilarityScorer(opts)
    a = "The cat sat on the mat."
    b = "A cat was sitting on a thick thick thick sofa."
    score, content = scorer.similarity_score(a, b)
    print(f"Similarity score: {score} ({content})")
