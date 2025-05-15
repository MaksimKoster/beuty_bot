FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/rag/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/rag/rag_fill.py ./rag_fill.py
COPY cosmetic.csv ./cosmetic.csv
COPY product_outputs ./product_outputs

CMD ["python", "rag_fill.py"]
