import json

count = 0

with open(
    "data/merged_corpus.jsonl",
    "r",
    encoding="utf-8"
) as f:

    for line in f:
        count += 1

print("Total passages:", count)

with open(
    "data/merged_corpus.jsonl",
    "r",
    encoding="utf-8"
) as f:

    print("\nFirst Record:\n")
    print(json.loads(next(f)))