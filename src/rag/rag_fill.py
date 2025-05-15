import json
import os
from json_repair import repair_json
from tqdm import tqdm 
from qdrant_client import QdrantClient
import pandas as pd 

def get_price(product_url):
    sub = df[df['product_url'] == product_url]
    prices = sub['regular_price'].values
    if len(prices) > 0:
        return str(sub['regular_price'].values[0])
    else:
        return str(250)


client = QdrantClient(url="http://127.0.0.1:6333")
client.set_model("sentence-transformers/all-MiniLM-L6-v2")
client.set_sparse_model("Qdrant/bm25")

if not client.collection_exists("beaty"):
    client.create_collection(
        collection_name="beaty",
        vectors_config=client.get_fastembed_vector_params(),
        sparse_vectors_config=client.get_fastembed_sparse_vector_params(),  
    )


df = pd.read_csv('cosmetic.csv')
directory_path = "product_outputs"

documents = []
metadata = []

for filename in tqdm(os.listdir(directory_path)):

    if filename.endswith(".json"):
        file_path = os.path.join(directory_path, filename)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data = repair_json(data, ensure_ascii=False)
            data = json.loads(data)
            data['price_in_rubles'] = get_price(data['product_url'])
            data['product_url'] = "https://goldapple.ru/" + data['product_url']
            documents.append(data.pop("description"))
            metadata.append(data)

client.add(
    collection_name="beaty",
    documents=documents,
    metadata=metadata,
    ids=tqdm(range(len(documents))),
)