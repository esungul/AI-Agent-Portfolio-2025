from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
with open("ai_corpus.txt", "r", encoding="utf-8") as f:
    lines = f.read().split("\n\n")
    corpus = [line.split(": ", 1)[1] for line in lines if line.startswith("Snippet")]

def encode_corpus(corpus):
    embeddings = model.encode(corpus, batch_size=8, show_progress_bar=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms  # Normalize
    print(f"Encoded {len(embeddings)} snippets, dimension: {embeddings.shape[1]}")
    print(f"Embedding norm for Snippet 4: {np.linalg.norm(embeddings[3]):.2f}")
    return embeddings

embeddings = encode_corpus(corpus[:100])
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "ai_corpus_index.bin")
