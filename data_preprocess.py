import json
from sentence_transformers import SentenceTransformer

def encode_data(model, json_file):
  with open(json_file, 'r') as file:
      data = json.load(file)
      texts = [item['text'] for item in data]
      embeddings = model.encode(texts, normalize_embeddings=True)
      
      finalized = []
      for i in range(len(data)):
        document = data[i]
        embedding = embeddings[i].tolist()
        json_str = {"id":document['id'], "text":document['text'], "embedding":embedding}
        finalized.append(json_str)
  return finalized

def preprocess(json_file, data):
  with open(json_file, "w") as file:
    json.dump(data, file, indent=4)

if __name__ == "__main__":
  model = SentenceTransformer("BAAI/bge-base-en-v1.5")

  data = encode_data(model, "documents.json")

  preprocess("preprocessed_documents.json", data)
  