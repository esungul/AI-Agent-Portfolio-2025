from transformers import DistilBertTokenizer, DistilBertModel,DistilBertForQuestionAnswering
import torch
import faiss
import numpy as np



######## Load models and corpus

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
qa_model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

index = faiss.read_index("ai_corpus_index.bin")
with open ("ai_corpus.txt", "r") as f:
    corpus = [line.split(": ", 1)[1] for line in f.read().split("\n\n") if line.startswith("Snippet")]

### Encode query 

def encode_query(question):
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
      emb = model(**inputs).last_hidden_state.mean(dim=1).numpy()
    return emb

def qa_chatbot(question, context):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = qa_model(**inputs)
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits)

    return tokenizer.decode(inputs['input_ids'][0][start_idx:end_idx])

def rag_chatbot(question):
    query_emb = encode_query(question)
    distances, indices = index.search(query_emb, k=3)  # TODO: Practice - Try k=5, compare context quality
    context = " ".join([corpus[i] for i in indices[0]])
    print(f"Question: {question}\nRetrieved context: {context[:100]}...")  # Debug
    answer = qa_chatbot(question, context)
    return answer

if __name__ == "__main__":
    questions = [
        "What is agentic AI?",
        "What are multimodal models?",
        "How does RAG work?",
        "What is a transformer model?",  # TODO: Practice - Add 1 more query
        "What is deep learning?"
    ]
    for q in questions:
        print(f"Answer: {rag_chatbot(q)}\n")

