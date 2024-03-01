import embedding.CLIP as CLIP
from pinecone import Pinecone

class PineconeVectorStore:
    def __init__(self, api_key: str, index_name: str):
        self.clip_model = CLIP.load_model()
        self.index = Pinecone(api_key=api_key).Index(index_name)

    def query(self, query: str):
        self.clip_embeddings = self.clip_model.encode(query).tolist()
        result = self.index.query(
            top_k=12, vector=self.clip_embeddings, include_metadata=True
        )

        result_matches = result["matches"]

        return result_matches
