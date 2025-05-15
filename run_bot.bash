docker build -f src/bot/Dockerfile -t beauty-bot .
docker run -p 8501:8501 beauty-bot