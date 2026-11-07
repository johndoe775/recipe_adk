import os

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

app = FastAPI(
    title="Recipe Embedding Service",
    version="1.0.0",
    description="Standalone embedding API for Pinecone-compatible recipe search.",
)

model = SentenceTransformer(MODEL_NAME)


class EmbedRequest(BaseModel):
    text: str
    normalize: bool = False


class BatchEmbedRequest(BaseModel):
    texts: list[str]
    normalize: bool = False


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "recipe-embedding",
        "model": MODEL_NAME,
    }


@app.get("/health")
def health():
    test_vector = model.encode("recipe health check", normalize_embeddings=True, convert_to_numpy=True)
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "dimension": int(len(test_vector)),
    }


@app.post("/embed")
def embed(payload: EmbedRequest):
    vector = model.encode(
        payload.text,
        normalize_embeddings=payload.normalize,
        convert_to_numpy=True,
    )
    return {
        "embedding": vector.tolist(),
        "dimension": int(len(vector)),
        "model": MODEL_NAME,
    }


@app.post("/batch-embed")
def batch_embed(payload: BatchEmbedRequest):
    vectors = model.encode(
        payload.texts,
        normalize_embeddings=payload.normalize,
        convert_to_numpy=True,
    )
    return {
        "embeddings": [vec.tolist() for vec in vectors],
        "count": len(payload.texts),
        "dimension": int(vectors.shape[1]),
        "model": MODEL_NAME,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8001")),
        reload=False,
    )
