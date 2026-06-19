from datasets import load_dataset

dataset = load_dataset(
    "MedRAG/textbooks"
)

print(dataset)

print("\nFIRST SAMPLE:\n")

print(dataset["train"][0])