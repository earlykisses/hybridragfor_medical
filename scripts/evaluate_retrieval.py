import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from retrieval.hybrid_retriever import HybridRetriever
from datasets import load_dataset
from tqdm import tqdm

print("Loading retriever...")

retriever = HybridRetriever()

print("Loading PubMedQA...")

dataset = load_dataset(
    "qiaojin/PubMedQA",
    "pqa_labeled"
)

# Start with 100 for testing
# Later change to 1000
NUM_SAMPLES = 100

recall_1_hits = 0
recall_5_hits = 0
recall_10_hits = 0

print(f"\nEvaluating on {NUM_SAMPLES} samples...\n")

for i in tqdm(range(NUM_SAMPLES)):

    sample = dataset["train"][i]

    question = sample["question"]
    pubid = str(sample["pubid"])

    retrieved = retriever.retrieve(
        question,
        top_k=10
    )

    retrieved_ids = [
        item[0]
        for item in retrieved
    ]

    # Recall@1
    if any(
        doc_id.startswith(pubid)
        for doc_id in retrieved_ids[:1]
    ):
        recall_1_hits += 1

    # Recall@5
    if any(
        doc_id.startswith(pubid)
        for doc_id in retrieved_ids[:5]
    ):
        recall_5_hits += 1

    # Recall@10
    if any(
        doc_id.startswith(pubid)
        for doc_id in retrieved_ids[:10]
    ):
        recall_10_hits += 1

recall_at_1 = recall_1_hits / NUM_SAMPLES
recall_at_5 = recall_5_hits / NUM_SAMPLES
recall_at_10 = recall_10_hits / NUM_SAMPLES

print("\n" + "=" * 60)
print("RETRIEVAL EVALUATION RESULTS")
print("=" * 60)

print(f"Recall@1  : {recall_at_1:.4f}")
print(f"Recall@5  : {recall_at_5:.4f}")
print(f"Recall@10 : {recall_at_10:.4f}")

print("=" * 60)
print(f"Samples Evaluated : {NUM_SAMPLES}")
print(f"Hits@1            : {recall_1_hits}")
print(f"Hits@5            : {recall_5_hits}")
print(f"Hits@10           : {recall_10_hits}")
print("=" * 60)