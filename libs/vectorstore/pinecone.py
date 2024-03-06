import io
import base64

from typing import List
from PIL.Image import Image as ImageFile
from pinecone import Pinecone, ServerlessSpec
from libs.embedding.clip.encoder import CLIPEncoder
from libs.embedding.bm25.encoder import Bm25Encoder


class PineconeIndex:
    def __init__(self, api_key: str, index_name: str, bm25_fit_corpus: List[str]):
        pinecone = Pinecone(api_key=api_key)

        if index_name not in pinecone.list_indexes().names():
            pinecone.create_index(
                index_name,
                dimension=512,
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region="us-west-2"),
            )

        self.index = pinecone.Index(index_name)
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

    def upsert_images(
        self, ids: List, images: List, meta_list: List, meta_dict: dict
    ) -> None:

        vectors = []

        # create sparse BM25 vectors
        sparse_embeds = self.sparse_embedding.encode_documents(
            texts=[text for text in meta_list]
        )

        # create vectors values
        dense_embeds = self.dense_embedding.encode(images)

        for _id, dense, sparse, meta in zip(
            ids, dense_embeds, sparse_embeds, meta_dict
        ):
            img_bytes = io.BytesIO()
            image = images[int(_id)]

            # Save the image to the in-memory stream in JPEG format
            image.save(img_bytes, format="JPEG")

            meta["image_b64"] = base64.b64encode(img_bytes.getvalue()).decode("utf-8")

            vectors.append(
                {"id": _id, "values": dense, "sparse_values": sparse, "metadata": meta}
            )

        self.index.upsert(vectors)
