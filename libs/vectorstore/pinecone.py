from typing import List
import streamlit as st

from pinecone import Pinecone
from PIL.Image import Image as ImageFile
from libs.embedding.clip.encoder import CLIPEncoder
from libs.embedding.bm25.encoder import Bm25Encoder


class PineconeIndex:
    def __init__(self, api_key: str, index_name: str, bm25_fit_corpus: List[str]):
        self.index = Pinecone(api_key=api_key).Index(index_name)

        self.dense_embedding = CLIPEncoder()
        self.sparse_embedding = Bm25Encoder(fit_corpus=bm25_fit_corpus)

    def hybrid_scale(
        self, dense: List, sparse: List, alpha: float
    ) -> tuple[List, List]:
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")

        hsparse = {
            "indices": sparse["indices"],
            "values": [v * (1 - alpha) for v in sparse["values"]],
        }

        hdense = [v * alpha for v in dense]

        return hdense, hsparse

    def hybrid_search(
        self,
        query: str,
        alpha: float,
        top_k: int = 12,
    ) -> List[dict]:

        dense = self.dense_embedding.encode(query)
        sparse = self.sparse_embedding.encode_queries(texts=query)

        vector, sparce_vector = self.hybrid_scale(dense, sparse, alpha)

        result = self.index.query(
            top_k=top_k,
            vector=vector,
            sparse_vector=sparce_vector,
            include_metadata=True,
        )

        return result["matches"]

    def image_search(
        self,
        image: ImageFile,
        top_k: int = 12,
    ):
        vector = self.dense_embedding.encode(image)
        result = self.index.query(
            top_k=top_k,
            vector=vector,
            include_metadata=True,
        )

        return result["matches"]

    def upsert_vectors(self, vectors: List[dict]):
        self.index.upsert(vectors)
