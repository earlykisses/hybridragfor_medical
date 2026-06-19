# scripts/check_corpus.py

count = 0

with open(
    "data/corpus.jsonl",
    "r",
    encoding="utf-8"
) as f:

    for _ in f:
        count += 1

print("Total passages:", count)