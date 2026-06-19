from datasets import load_dataset
import json

print("Loading StatPearls...")

dataset = load_dataset(
    "HanyangMed/statpearls"
)

output_file = "data/statpearls_corpus.jsonl"

count = 0

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    for row in dataset["train"]:

        record = {
            "id": f"statpearls_{row['id']}",
            "text": row["contents"],
            "source": "StatPearls"
        }

        f.write(
            json.dumps(record)
            + "\n"
        )

        count += 1

print(f"\nSaved {count} passages")
print(f"Output: {output_file}")