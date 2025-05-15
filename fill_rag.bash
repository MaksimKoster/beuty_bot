docker build -f src/rag/Dockerfile -t qdrant-loader .
docker run --rm --network=host qdrant-loader