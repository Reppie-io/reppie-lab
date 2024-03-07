import io
import base64

from typing import List
from PIL.Image import Image as ImageFile
from pinecone import Pinecone as PineconeClient, ServerlessSpec, Index
from libs.embedding.clip.encoder import CLIPEncoder
from libs.embedding.bm25.encoder import Bm25Encoder
from libs.embedding.all_minilm.encoder import AllMiniLMEncoder


class Pinecone:
    def __init__(self, api_key: str, index_name: str, bm25_params_path: str):
        self.index_name = index_name
        self.pinecone = PineconeClient(api_key=api_key)
        self.clip_encoder = CLIPEncoder()
        self.bm25_encoder = Bm25Encoder(params_path=bm25_params_path)
        self.all_minilm_encoder = AllMiniLMEncoder()

    def create_index(
        self, index_name: str, dimension: int, metric: str = "dotproduct"
    ) -> None:
        self.pinecone.create_index(
            index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-west-2"),
        )

    def index(self) -> Index:
        return self.pinecone.Index(self.index_name)

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
        is_image_search: bool = False,
    ) -> List[dict]:

        if is_image_search:
            dense = self.clip_encoder.encode(input=query)
        else:
            dense = self.all_minilm_encoder.encode(texts=query)

        sparse = self.bm25_encoder.encode_queries(texts=query)

        vector, sparce_vector = self.hybrid_scale(dense, sparse, alpha)

        result = self.index().query(
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
        vector = self.clip_encoder.encode(image)
        result = self.index().query(
            top_k=top_k,
            vector=vector,
            include_metadata=True,
        )

        return result["matches"]

    def upsert_images(
        self,
        ids: List,
        images: List[str],
        keyword_texts: List[str],
        metadata: List[dict],
    ) -> None:

        vectors = []

        # create sparse BM25 vectors
        sparse_embeds = self.bm25_encoder.encode_documents(
            texts=[keyword_text for keyword_text in keyword_texts]
        )

        # create vectors values
        dense_embeds = self.clip_encoder.encode(input=images)

        for _id, dense, sparse, meta in zip(ids, dense_embeds, sparse_embeds, metadata):
            img_bytes = io.BytesIO()
            image = images[int(_id)]

            # Save the image to the in-memory stream in JPEG format
            image.save(img_bytes, format="JPEG")

            meta["image_b64"] = base64.b64encode(img_bytes.getvalue()).decode("utf-8")

            vectors.append(
                {"id": _id, "values": dense, "sparse_values": sparse, "metadata": meta}
            )

        self.index().upsert(vectors)

    def upsert_texts(
        self,
        ids: List,
        texts: List[str],
        keyword_texts: List[str],
        metadata: List[dict],
        chunk_size: int = 1000,
    ):
        vectors = []

        for _id, text, keyword_text, meta in zip(ids, texts, keyword_texts, metadata):
            print(f"Upserting {_id}, text: {text[:50]}...")

            # create sparse BM25 vectors
            sparse_values = self.bm25_encoder.encode_documents(texts=keyword_text)

            # text_splitter = RecursiveCharacterTextSplitter(
            #     chunk_size=chunk_size,
            #     chunk_overlap=50,
            #     length_function=len,
            #     is_separator_regex=False,
            # )

            # chunks = text_splitter.split_text(text)
            # print(f"Text split into {len(chunks)} chunks")

            # for idx, chunk in enumerate(chunks):
            #     print(
            #         f"Upserting chunk {idx+1} of {len(chunks)}, chunk: {chunk[:50]}..."
            #     )

            # create vectors values
            values = self.all_minilm_encoder.encode(texts=text)
            # meta_chunk = meta
            # meta_chunk["text"] = chunk

            # print(f"metadata: {meta_chunk}")

            vectors.append(
                {
                    "id": _id,
                    "values": values,
                    "sparse_values": sparse_values,
                    "metadata": meta,
                }
            )

        self.index().upsert(vectors)
