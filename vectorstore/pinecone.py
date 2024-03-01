from typing import List
import streamlit as st

from pinecone import Pinecone
from PIL.Image import Image as ImageFile
from embedding.CLIP.sentence_transformer import CLIPSentenceTransformer


class PineconeIndex:
    def __init__(self, index_name: str):
        self.index = Pinecone(api_key=st.secrets["PINECONE_API_KEY"]).Index(index_name)
        self.embedding = CLIPSentenceTransformer(
            model_name="sentence-transformers/clip-ViT-B-32"
        )

    def semantic_search(
        self,
        query: str | ImageFile,
        top_k: int = 12,
    ):

        vector = self.embedding.encode(query)
        result = self.index.query(top_k=top_k, vector=vector, include_metadata=True)

        return result["matches"]

    def upsert_vectors(self, vectors: List[dict]):
        self.index.upsert(vectors)
