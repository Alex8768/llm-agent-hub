from sentence_transformers import SentenceTransformer

class LocalEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]):
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str):
        return self.embed([text])[0]
    
from sentence_transformers import SentenceTransformer
EMB_MODEL = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="data/.hf_cache",
)