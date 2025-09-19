from transformers import DistilBertTokenizer, DistilBertModel
import torch
import faiss
import numpy as np

# Load DistilBERT for embeddings
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Load corpus
with open("ai_corpus.txt", "r", encoding="utf-8") as f:
    lines = f.read().split("\n\n")
    corpus = [line.split(": ", 1)[1] for line in lines if line.startswith("Snippet")]  # Extract text after "Snippet X:"

# Encode corpus into embeddings
def encode_corpus(corpus, batch_size=16):  # TODO: Practice - Try batch_size=8; how does it affect speed?
    embeddings = []
    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling of last hidden state
            batch_emb = outputs.last_hidden_state.mean(dim=1).numpy()  # TODO: Practice - Try max pooling instead
        embeddings.append(batch_emb)
    return np.vstack(embeddings)

# Create FAISS index
embeddings = encode_corpus(corpus[:100])  # Limit to 100 snippets
dimension = embeddings.shape[1]  # E.g., 768 for DistilBERT
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index (optional)
faiss.write_index(index, "ai_corpus_index.bin")

# Test on subset
print(f"Encoded {len(embeddings)} snippets, dimension: {dimension}")
print("Sample embedding:", embeddings[0, :5])  # TODO: Practice - Print shape of embeddings
