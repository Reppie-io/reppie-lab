from typing import List
import torch
from sentence_transformers import SentenceTransformer


class AllMiniLMEncoder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(
            model_name, device=device, cache_folder=".cache"
        )

    def encode(self, texts: str | List[str]) -> List:
        return self.model.encode(texts).tolist()
