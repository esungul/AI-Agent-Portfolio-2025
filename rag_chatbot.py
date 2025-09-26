from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
import re

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
qa_model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
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
    emb = embed_model.encode([question])[0]
    norm = np.linalg.norm(emb)
    emb = emb / norm if norm > 0 else emb
    print(f"Query embedding shape: {emb.shape}, norm: {norm:.2f}")
    return emb[np.newaxis, :]

def extend_answer_to_sentence_boundary(full_context, partial_answer):
    """Extend a partial answer to a complete sentence using the full context"""
    # Find the partial answer in the context
    start_pos = full_context.lower().find(partial_answer.lower())
    if start_pos == -1:
        return partial_answer  # Return original if not found

    # Look for sentence end after the partial answer
    sentence_end_chars = ['.', '!', '?', '\n']

    # Search for the next sentence boundary
    for i in range(start_pos + len(partial_answer), len(full_context)):
        if full_context[i] in sentence_end_chars:
            extended = full_context[start_pos:i+1].strip()
            # Ensure we have a complete thought (not just a fragment)
            if len(extended.split()) > len(partial_answer.split()) + 2:  # At least 2 more words
                return extended

    # If no sentence boundary found, try to extend by words until we have a complete phrase
    words = full_context[start_pos:].split()
    partial_words = partial_answer.split()

    # Add more words to make it more complete (up to 10 additional words)
    if len(words) > len(partial_words):
        extended_words = words[:min(len(partial_words) + 8, len(words))]
        extended = ' '.join(extended_words)

        # Add period if missing
        if not extended.endswith(('.', '!', '?')):
            extended += '.'
        return extended

    return partial_answer

def get_best_answer_span(inputs, outputs, context_text):
    """Improved answer extraction that considers multiple possible spans"""
    start_scores = torch.softmax(outputs.start_logits, dim=1)
    end_scores = torch.softmax(outputs.end_logits, dim=1)

    # Get top 3 start and end positions
    top_k = 3
    start_indices = torch.topk(start_scores, top_k, dim=1).indices[0]
    end_indices = torch.topk(end_scores, top_k, dim=1).indices[0]

    best_answer = ""
    best_confidence = 0
    best_start_idx = 0
    best_end_idx = 0

    # Try different combinations of start and end positions
    for start_idx in start_indices:
        for end_idx in end_indices:
            if start_idx < end_idx and start_idx > 0:  # Valid span
                confidence = (start_scores[0, start_idx] * end_scores[0, end_idx]).item()

                # Prefer answers with good confidence
                answer_length = end_idx - start_idx
                if confidence > best_confidence and answer_length <= 25:  # Reasonable length limit
                    answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
                    answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True)

                    if answer_text.strip():  # Non-empty answer
                        best_answer = answer_text
                        best_confidence = confidence
                        best_start_idx = start_idx
                        best_end_idx = end_idx

    return best_answer, best_confidence, best_start_idx, best_end_idx

def qa_chatbot(question, context):
    inputs = tokenizer(question, context, return_tensors="pt", truncation="only_second", max_length=512)
    print(f"Input token count: {len(inputs['input_ids'][0])}")

    with torch.no_grad():
        outputs = qa_model(**inputs)

    # Use improved answer extraction
    answer, confidence, start_idx, end_idx = get_best_answer_span(inputs, outputs, context)

    print(f"Best answer confidence: {confidence:.2f}")
    print(f"Raw extracted answer: '{answer}'")

    if confidence < 0.2 or not answer:
        return "I'm not sure about the answer based on the available information."

    # Extend to complete sentence
    if confidence > 0.3:  # Only extend if we have reasonable confidence
        extended_answer = extend_answer_to_sentence_boundary(context, answer)
        if extended_answer != answer:
            print(f"Extended answer: '{extended_answer}'")
            answer = extended_answer

    # Basic cleaning and formatting
    answer = answer.strip()

    # Ensure it starts properly for definition questions
    if question.lower().startswith('what is'):
        if not answer.lower().startswith(('it is', 'it\'s', 'this is', 'a ', 'an ', 'the ')):
            answer = answer[0].lower() + answer[1:]  # Ensure lowercase start
            if not answer.startswith('a ') and not answer.startswith('an ') and not answer.startswith('the '):
                answer = "a " + answer

    # Capitalize first letter
    if answer and answer[0].islower():
        answer = answer[0].upper() + answer[1:]

    # Ensure it ends with punctuation
    if answer and not answer.endswith(('.', '!', '?')):
        answer += '.'

    return answer

def rag_chatbot(question):
    query_emb = encode_query(question)
    distances, indices = index.search(query_emb, k=3)
    print(f"Retrieved indices: {indices[0]}, distances: {distances[0]}")

    context_parts = []
    total_tokens = 0
    max_tokens = 500  # Increased context window

    for i in indices[0]:
        if i < len(corpus):
            snippet = corpus[i]
            print(f"Snippet {i+1}: {snippet[:100]}...")
            tokens = tokenizer.encode(snippet, add_special_tokens=False)
            allowed_tokens = tokens[:150]  # Increased per-snippet limit

            if total_tokens + len(allowed_tokens) < max_tokens:
                snippet_text = tokenizer.decode(allowed_tokens, skip_special_tokens=True)
                context_parts.append(snippet_text)
                total_tokens += len(allowed_tokens)
            else:
                break

    if not context_parts:
        print("No relevant context found")
        return "No relevant context found"

    context = " ".join(context_parts)
    print(f"Question: {question}")
    print(f"Retrieved context length: {len(context)} characters")
    print(f"Context preview: {context[:200]}...")

    return qa_chatbot(question, context.strip())

if __name__ == "__main__":
    questions = [
        "What is agentic AI?",
        "How does machine learning work?",
    ]
    for q in questions:
        print(f"\n{'='*60}")
        print(f"Question: {q}")
        answer = rag_chatbot(q)
        print(f"Final Answer: {answer}")
        print(f"{'='*60}")
