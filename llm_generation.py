import time
def llm_generate(llm, I, query, id_to_doc):
  start = time.perf_counter()
  documents = [id_to_doc[id]['text'] for id in I[0]]
  documents = "".join(documents)
  end = time.perf_counter()
  document_retrieval = (end - start)*1000

  start = time.perf_counter()
  prompt = query + "Top Documents:" + documents
  end = time.perf_counter()

  prompt_augmentation = (end - start)*1000

  # print("Retrieving Documents Done")
  start = time.perf_counter()
  output = llm(prompt=prompt, max_tokens=256, echo=False)
  end = time.perf_counter()

  generation = (end - start)*1000
  return output["choices"][0]["text"], document_retrieval, prompt_augmentation, generation
# def llm_generate(llm, I, query, id_to_doc):
#   prompt = query + "Top Documents:" 
#   for i in range(len(I[0])):
#     id = I[0][i]
#     prompt += id_to_doc[id]['text']
#     # print(f"Doc {id}: {id_to_doc[id]['text']}")
#   # end = time.perf_counter()
#   # elapse = (end - start)*1000

#   # print("Retrieving Documents Done")
#   # start_generation = time.perf_counter()
#   output = llm(prompt=prompt, max_tokens=256, stop=["\n"],echo=False)
#   return output["choices"][0]["text"]