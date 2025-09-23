from transformers import DistilBertTokenizer, DistilBertModel
import torch
import faiss
import numpy as np

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

with open("ai_corpus.txt", "r", encoding="utf-8") as f:
    lines = f.read().split("\n\n")
    corpus = [line.split(": ", 1)[1] for line in lines if line.startswith("Snippet")]

def encode_corpus(corpus, batch_size=8):
    embeddings = []
    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_emb = outputs.last_hidden_state.mean(dim=1).numpy()
        embeddings.append(batch_emb)
    embeddings = np.vstack(embeddings)
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    print(f"Encoded {len(embeddings)} snippets, dimension: {embeddings.shape[1]}")
    print(f"Embedding norm for Snippet 4: {np.linalg.norm(embeddings[3]):.2f}")  # 0-based index
    return embeddings

embeddings = encode_corpus(corpus[:100])
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "ai_corpus_index.bin")
