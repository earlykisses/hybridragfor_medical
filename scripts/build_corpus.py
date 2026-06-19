from datasets import load_dataset
import json
from tqdm import tqdm

print("Loading PubMedQA...")

dataset = load_dataset(
    "qiaojin/PubMedQA",
    "pqa_labeled"
)

output_file = "data/corpus.jsonl"

count = 0

with open(output_file, "w", encoding="utf-8") as f:

    for sample in tqdm(dataset["train"]):

        pubid = sample["pubid"]

        contexts = sample["context"]["contexts"]

        for idx, text in enumerate(contexts):

            record = {
                "id": f"{pubid}_{idx}",
                "text": text,
                "source": "PubMedQA"
            }

            f.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )

            count += 1

print(f"\nSaved {count} passages")
print(f"Corpus stored at: {output_file}")