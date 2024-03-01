import torch
from sentence_transformers import SentenceTransformer


class CLIPSentenceTransformer:
    def __init__(self, model_name: str = "sentence-transformers/clip-ViT-B-32"):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(
            model_name, device=device, cache_folder=".cache"
        )

    def encode(self, text: str | bytes):
        return self.model.encode(text).tolist()
