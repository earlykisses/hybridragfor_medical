from datasets import load_dataset
import json

print("Loading Textbooks...")

dataset = load_dataset(
    "MedRAG/textbooks"
)

output_file = "data/textbooks_corpus.jsonl"

count = 0

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    for row in dataset["train"]:

        record = {
            "id": row["id"],
            "text": row["contents"],
            "source": "Textbooks"
        }

        f.write(
            json.dumps(record)
            + "\n"
        )

        count += 1

print(f"\nSaved {count} passages")
print(f"Output: {output_file}")