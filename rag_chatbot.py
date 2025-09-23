import torch
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertForQuestionAnswering
import faiss
import numpy as np

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
qa_model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
try:
    index = faiss.read_index("ai_corpus_index.bin")
    print(f"Index loaded with {index.ntotal} vectors")
except:
    print("Index missing; run encode_corpus.py")
    exit(1)
with open("ai_corpus.txt", "r", encoding="utf-8") as f:
    corpus = [line.split(": ", 1)[1] for line in f.read().split("\n\n") if line.startswith("Snippet")]
print(f"Corpus size: {len(corpus)} snippets")

def encode_query(question):
    inputs = tokenizer(question, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        emb = model(**inputs).last_hidden_state.mean(dim=1).numpy()
        norm = np.linalg.norm(emb)
        emb = emb / norm if norm > 0 else emb  # Normalize
    print(f"Query embedding shape: {emb.shape}, norm: {norm:.2f}")
    return emb

def qa_chatbot(question, context):
    inputs = tokenizer(question, context, return_tensors="pt", truncation="only_second", max_length=512)
    print(f"Input token count: {len(inputs['input_ids'][0])}")
    with torch.no_grad():
        outputs = qa_model(**inputs)
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits) + 1
    print(f"Logits: start_idx={start_idx}, end_idx={end_idx}")
    if start_idx >= end_idx or start_idx == 0:
        return "No answer found"
    return tokenizer.decode(inputs['input_ids'][0][start_idx:end_idx])

def rag_chatbot(question):
    query_emb = encode_query(question)
    distances, indices = index.search(query_emb, k=2)  # TODO: Practice - Try k=1
    print(f"Retrieved indices: {indices[0]}, distances: {distances[0]}")
    context = ""
    for i in indices[0]:
        if i < len(corpus):
            snippet = corpus[i]
            print(f"Snippet {i+1}: {snippet[:50]}...")
            if len(tokenizer.encode(context + " " + snippet, add_special_tokens=False)) < 100:
                context += " " + snippet
            else:
                break
    if not context:
        print("No relevant context found")
        return "No relevant context found"
    print(f"Question: {question}\nRetrieved context: {context[:100]}...")
    return qa_chatbot(question, context.strip())

if __name__ == "__main__":
    question = "What is agentic AI?"
    print(f"Answer: {rag_chatbot(question)}\n")
    questions = [
        "What is agentic AI?"
    ]
    for q in questions:
        print(f"Answer: {rag_chatbot(q)}\n")
