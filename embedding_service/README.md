# Recipe Embedding Service

This is a standalone Render deployment for the Pinecone-compatible embedding model.

## Model

- sentence-transformers/all-MiniLM-L6-v2

## Endpoints

- GET /
- GET /health
- POST /embed
- POST /batch-embed

## Example request

curl -X POST https://YOUR_RENDER_URL/embed \
  -H "Content-Type: application/json" \
  -d '{"text":"chicken curry","normalize":false}'

## Render start command

uvicorn app:app --host 0.0.0.0 --port $PORT
