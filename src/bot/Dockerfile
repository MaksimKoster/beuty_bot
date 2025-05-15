FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/bot/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/bot/bot.py ./bot.py
COPY src/bot/logo.png ./logo.png

CMD ["streamlit", "run", "bot.py"]
