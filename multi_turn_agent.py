from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer
import torch
import re  # For simple city extraction in tool

# Initialize models
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
qa_model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

# Load a simple context file (create or reuse ai_corpus.txt)
try:
    with open("ai_context.txt", "r", encoding="utf-8") as f:
        context = f.read().strip()
    print(f"Context loaded: {context[:100]}...")
except FileNotFoundError:
    print("Context file missing; creating a default one")
    context = (
        "Agentic AI is a class of artificial intelligence that focuses on autonomous systems that can make decisions. "
        "It works by adapting and learning from data inputs, making decisions based on continuous learning. "
        "Multimodal models integrate multiple data types, such as text and images, to perform tasks."
    )
    with open("ai_context.txt", "w", encoding="utf-8") as f:
        f.write(context)
    print(f"Default context created: {context[:100]}...")

# Conversation history
conversation_history = []

# Simple mock tool: Get weather (returns hardcoded data for beginner simplicity)
def get_weather(question):
    # Extract city using basic regex (e.g., "in New York" -> "New York")
    match = re.search(r'(?:in|for|at)\s+([A-Za-z\s]+)', question, re.IGNORECASE)
    city = match.group(1).strip() if match else "Unknown"
    # Mock data; in real projects, replace with API call
    weather_data = {
        "New York": "Sunny with 75°F.",
        "London": "Rainy with 60°F.",
        "Tokyo": "Cloudy with 70°F.",
        "Unknown": "Sorry, couldn't determine the city."
    }
    return f"The weather in {city} is {weather_data.get(city, 'unknown')}"

def multi_turn_agent(question):
    global conversation_history
    if not question.strip():
        return "Please provide a valid question."
    
    # Tool trigger: Check for "weather" keyword
    tool_result = ""
    if "weather" in question.lower():
        tool_result = get_weather(question)
        print(f"Tool called: {tool_result}")
    
    # Build context with history (last 2 interactions) and tool result
    history_context = " ".join([f"Q: {q} A: {a}" for q, a in conversation_history[-2:]])
    full_input = f"{history_context} {tool_result} Q: {question}" if history_context or tool_result else question
    
    # Tokenize input with context
    inputs = tokenizer(full_input, context, return_tensors="pt", truncation="only_second", max_length=512)
    token_count = len(inputs['input_ids'][0])
    print(f"Input token count: {token_count}")
    if token_count > 512:
        return "Input too long; please simplify the question."
    
    # Generate answer
    with torch.no_grad():
        outputs = qa_model(**inputs)
    start_scores = torch.softmax(outputs.start_logits, dim=1)
    end_scores = torch.softmax(outputs.end_logits, dim=1)
    start_idx = torch.argmax(start_scores, dim=1).item()
    end_idx = torch.argmax(end_scores, dim=1).item()
    
    # Extract answer
    if start_idx < end_idx and start_idx > 0:
        answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
        confidence = (start_scores[0, start_idx] * end_scores[0, end_idx]).item()
        print(f"Logits: start_idx={start_idx}, end_idx={end_idx}, confidence={confidence:.2f}")
    else:
        answer = "I'm not sure about the answer based on the available information."
        confidence = 0
        print("No valid answer span found")
    
    # If tool was used, prepend tool result to answer for clarity
    if tool_result:
        answer = f"{tool_result} {answer}".strip()
    
    # Capitalize and punctuate answer
    if answer and answer[0].islower():
        answer = answer[0].upper() + answer[1:]
    if answer and not answer.endswith(('.', '!', '?')):
        answer += '.'
    
    # Update conversation history (limit to 2 pairs)
    conversation_history.append((question, answer))
    if len(conversation_history) > 2:
        conversation_history = conversation_history[-2:]
    
    print(f"Question: {question}")
    print(f"Context used: {context[:100]}...")
    print(f"History: {conversation_history}")
    return answer

if __name__ == "__main__":
    questions = [
        "What is agentic AI?",
        "How does it work?",  # Multi-turn
        "What's the weather in New York?",  # Tool trigger
        "Is it raining?",  # Multi-turn with tool context
        "What are multimodal models?"
    ]
    for q in questions:
        print(f"\n{'='*60}")
        print(f"Question: {q}")
        answer = multi_turn_agent(q)
        print(f"Final Answer: {answer}")
        print(f"{'='*60}")
