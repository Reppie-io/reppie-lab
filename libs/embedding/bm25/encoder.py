from typing import List
from pinecone_text.sparse import BM25Encoder, SparseVector


class Bm25Encoder:
    def __init__(self, params_path: str):
        self.model = BM25Encoder()
        self.model.load(params_path)

    def encode_queries(self, texts: str | List[str]) -> List[SparseVector]:
        return self.model.encode_queries(texts=texts)

    def encode_documents(
        self, texts: List[str]
    ) -> List[SparseVector]:
        return self.model.encode_documents(texts=texts)
