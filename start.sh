#!/bin/bash
set -e
cd ~/projects/privatedoc-agent
source .venv/bin/activate

echo "Starting Qdrant..."
docker compose -f infra/docker-compose.yml up -d qdrant

echo "Starting llama-server..."
fuser -k 8080/tcp 2>/dev/null || true
python -m llama_cpp.server \
  --model models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --port 8080 \
  --n_gpu_layers 24 \
  --n_ctx 4096 &

echo "Waiting for model to load (50s)..."
sleep 50

echo "Starting FastAPI..."
fuser -k 8000/tcp 2>/dev/null || true
uvicorn api.main:app --port 8000
