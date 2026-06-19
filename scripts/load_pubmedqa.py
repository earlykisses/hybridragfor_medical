from datasets import load_dataset

dataset = load_dataset(
    "qiaojin/PubMedQA",
    "pqa_labeled"
)

print(dataset)
print("\nFIRST SAMPLE:\n")
print(dataset["train"][0])