import wikipedia
import re

# List of AI-related topics
topics = [
    "Artificial intelligence", "Agentic AI", "Multimodal model", "Retrieval-Augmented Generation",
    "Large language model", "Transformer model", "Deep learning", "Computer vision"
]

# Collect snippets
corpus = []
for topic in topics:
    try:
        page = wikipedia.page(topic, auto_suggest=False)
        # Split content into paragraphs, take first 2-3
        paragraphs = page.content.split("\n\n")[:3]
        # Clean text (remove headings, refs)
        cleaned = [re.sub(r"==.*?==|[\[\]\d+]", "", p).strip() for p in paragraphs if len(p) > 50]
        corpus.extend(cleaned)
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Disambiguation for {topic}: {e.options[:2]}")
    except wikipedia.exceptions.PageError:
        print(f"No page for {topic}")

# Save to file
with open("ai_corpus.txt", "w", encoding="utf-8") as f:
    for i, snippet in enumerate(corpus[:100], 1):  # Limit to 100
        f.write(f"Snippet {i}: {snippet}\n\n")

print(f"Collected {len(corpus)} snippets")
