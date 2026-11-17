import json, time, faiss
from sentence_transformers import SentenceTransformer
import ollama

# 1. one MedQA question (from Hugging Face)
from datasets import load_dataset
ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
q = ds[0]
question, options, answer = q["question"], q["options"], q["answer_idx"]

# 2. one fake clinical document (no real patient data)
doc = ("Discharge summary. 58-year-old male admitted with chest pain. "
       "ECG showed ST elevation. Treated with aspirin and PCI. "
       "Discharged on day 3 with follow-up in cardiology clinic.")

# 3. embed + retrieve
emb = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.IndexFlatL2(384)
index.add(emb.encode([doc]))
_, hit = index.search(emb.encode([question]), k=1)
context = doc

# 4. prompt the model
opts = "\n".join(f"{k}. {v}" for k, v in options.items())
prompt = (f"Use the document to help answer the question.\n\n"
          f"Document:\n{context}\n\nQuestion: {question}\n{opts}\n\n"
          f"Answer with the letter only.")

t = time.time()
r = ollama.chat(model="llama3:8b", messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0, "num_predict": 64})
print("Model said:", r["message"]["content"].strip())
print("Correct answer:", answer)
print(f"Time: {time.time() - t:.1f} s")