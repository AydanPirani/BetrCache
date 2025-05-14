from openai import OpenAI
from src.config import GPTOptions

MAX_SCORE = 100

class SimilarityScorer:
    def __init__(self, options: GPTOptions):
        self.client = OpenAI(api_key=options.api_key)
        self.model = options.model

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
        score, reason = content.split(";")
        return float(score.strip())/MAX_SCORE, reason


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
