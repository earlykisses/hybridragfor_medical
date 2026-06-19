from datasets import load_dataset

print("Loading StatPearls...")

dataset = load_dataset(
    "HanyangMed/statpearls"
)

print("\nDATASET INFO:\n")
print(dataset)

print("\nFIRST SAMPLE:\n")

print(dataset["train"][0])