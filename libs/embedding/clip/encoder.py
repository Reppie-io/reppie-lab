from typing import List
import torch
from PIL.Image import Image as ImageFile
from sentence_transformers import SentenceTransformer


class CLIPEncoder:
    def __init__(self, model_name: str = "sentence-transformers/clip-ViT-B-32"):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(
            model_name, device=device, cache_folder=".cache"
        )

    def encode(self, text: str | ImageFile) -> List:
        return self.model.encode(text).tolist()
