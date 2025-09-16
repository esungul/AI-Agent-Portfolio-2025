# Basic QA function skeleton
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer
import torch

# Load pre-trained DistilBERT model and tokenizer for QA
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")

# Basic QA function skeleton
def qa_chatbot(question, context):
    """
    Takes a question and context, tokenizes them, and returns tokenized inputs.
    Args:
        question (str): User's question (e.g., "What is agentic AI?")
        context (str): Context text containing the answer
    Returns:
        dict: Tokenized inputs (input_ids, attention_mask)
    """
    # Tokenize question and context
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True)
    
    # Explore inputs attributes
    print("Inputs keys:", inputs.keys())
    print("Input IDs shape:", inputs['input_ids'].shape)
    print("Attention mask shape:", inputs['attention_mask'].shape)
    if 'token_type_ids' in inputs:
        print("Token type IDs shape:", inputs['token_type_ids'].shape)
    else:
        print("Token type IDs: Not included (DistilBERT typically omits)")
    
    # Decode input_ids to verify text
    decoded_text = tokenizer.decode(inputs['input_ids'][0])
    print("Decoded input text:", decoded_text)
    
    # Show individual tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    print("Tokens:", tokens)
    
    return inputs

# Test the skeleton
if __name__ == "__main__":
    question = "What is agentic AI?"
    context = "Agentic AI refers to systems that autonomously make decisions and perform tasks in 2025."
    result = qa_chatbot(question, context)
    print("Returned inputs:", result)    return inputs
