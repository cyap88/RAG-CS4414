import faiss
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from data_preprocess import encode_data, preprocess
from vector_db import build_index, search, build_ivf_index
from encode import encode_query
from llm_generation import llm_generate
import sys

def main(args):
  model = SentenceTransformer("BAAI/bge-base-en-v1.5")
  # model = SentenceTransformer("BAAI/bge-m3")

  llm = Llama(
    model_path="tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf",
    n_ctx=2048,
    embedding=False,   
    logits_all=False,
    verbose = False,
  )
  # llm = Llama(
  #   model_path="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
  #   n_ctx=2048,
  #   embedding=False,   
  #   logits_all=False,
  #   verbose = False,
  # )
  print("creating preprocessed_documents.json")
  # data = encode_data(model, "documents.json")

  # preprocess("preprocessed_documents1.json", data)

  print("created preprocessed_documents.json")
  #preprocessed_documents.json is the 768 one
  index, embeddings, data, id_to_doc = build_index("preprocessed_documents1.json")
  # index, embeddings, data, id_to_doc = build_ivf_index("preprocessed_documents1.json")
  _, d = embeddings.shape

  while True:
    query = input("Query: ")
    if query.lower() == 'exit':
      break

    query_embedding = encode_query(model, query)
    query_embedding = query_embedding.reshape(1, d) # Shape: (1, 768)
    D, I = search(index, query_embedding, int(args[0]))
    output, _, _, _ = llm_generate(llm, I, query, id_to_doc)
    print(output)
  
  del llm
  del model

if __name__ == "__main__":
  main(sys.argv[1:])
  