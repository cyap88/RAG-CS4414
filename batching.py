import faiss
from sentence_transformers import SentenceTransformer
from vector_db import build_index, search
from encode import encode_query
import time
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
  model = SentenceTransformer("BAAI/bge-base-en-v1.5")
  
  # data = encode_data(model, "documents.json")
  # preprocess("preprocessed_documents.json", data)
  index, embeddings, data, id_to_doc = build_index("preprocessed_documents1.json")
  _, d = embeddings.shape
  
  latencies = []
  throughput = []

  query = "what are the benefits of fossil fuels"
  k = 3
  print("---Running performance test---")
  batches = [1, 4, 8, 16, 32, 64, 128]
  query_embedding = encode_query(model, query)

  for batch in batches:
    latency = 0
    query_batch = np.tile(query_embedding, (batch, 1)).astype('float32')

    for _ in range(20):
      # query_embedding = query_embedding.reshape(d,) 
      # query_embedding = query_embedding.flatten().astype('float32')
      
      start = time.perf_counter()
      D, I = search(index, query_batch, k)
      end = time.perf_counter()

      latency += (end-start)
    latency /= 20
    latencies.append(latency)
    throughput.append(batch/latency)

  del model

  print(f'Latencies = {latencies}')
  print(f'Throughput = {throughput}')
  print(f'Batches = {batches}')

  plt.figure(figsize=(12, 12))
  plt.subplot(2, 2, 1)
  plt.plot(batches, latencies)
  plt.title("Latencies per Batch")
  plt.xlabel("Batch")
  plt.ylabel("Latency (s)")

  plt.subplot(2, 2, 2)
  plt.plot(batches, throughput)
  plt.title("Throughput per Batch")
  plt.xlabel("Batch)")
  plt.ylabel("Throughput")

  plt.show()

  