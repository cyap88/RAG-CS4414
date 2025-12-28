import json
from sentence_transformers import SentenceTransformer

def encode_query(model, query):
  
  embedding = model.encode([query], normalize_embeddings=True)
  
  # embedding = embedding.flatten()
  return embedding

if __name__ == "__main__":
  model = SentenceTransformer("BAAI/bge-base-en-v1.5")

  data = encode_query(model, "does human hair stop squirrels?")
  print(data)
  print(data.shape)
  
  # preprocess("preprocessed_documents.json", data)
  