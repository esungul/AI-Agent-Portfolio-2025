with open("ai_corpus.txt", "r", encoding="utf-8") as f:
    lines = f.read().split("\n\n")
    snippets = [line.split(": ", 1)[1] for line in lines if line.startswith("Snippet")]

# Remove duplicates
unique_snippets = []
seen = set()
for i, snippet in enumerate(snippets, 1):
    if snippet not in seen:
        unique_snippets.append(f"Snippet {len(unique_snippets) + 1}: {snippet}")
        seen.add(snippet)

with open("ai_corpus.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(unique_snippets))
print(f"Saved {len(unique_snippets)} unique snippets")
