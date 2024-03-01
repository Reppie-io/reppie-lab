import torch
from sentence_transformers import SentenceTransformer


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return SentenceTransformer(
        "sentence-transformers/clip-ViT-B-32", device=device, cache_folder="./cache"
    )
