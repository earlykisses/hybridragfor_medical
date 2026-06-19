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

from datasets import load_dataset
from tqdm import tqdm

from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever


NUM_SAMPLES = 100


def evaluate_retriever(
    retriever,
    name
):

    print(f"\nEvaluating {name}...")

    recall_1 = 0
    recall_5 = 0
    recall_10 = 0

    mrr = 0.0

    samples = dataset["train"].select(
        range(NUM_SAMPLES)
    )

    for sample in tqdm(samples):

        question = sample["question"]
        pubid = str(
            sample["pubid"]
        )

        results = retriever.retrieve(
            question,
            top_k=10
        )

        retrieved_ids = [
            item[0]
            for item in results
        ]

        # Recall@1
        if any(
            doc_id.startswith(pubid)
            for doc_id in retrieved_ids[:1]
        ):
            recall_1 += 1

        # Recall@5
        if any(
            doc_id.startswith(pubid)
            for doc_id in retrieved_ids[:5]
        ):
            recall_5 += 1

        # Recall@10
        if any(
            doc_id.startswith(pubid)
            for doc_id in retrieved_ids[:10]
        ):
            recall_10 += 1

        # MRR
        rank_found = None

        for rank, doc_id in enumerate(
            retrieved_ids,
            start=1
        ):

            if doc_id.startswith(pubid):

                rank_found = rank
                break

        if rank_found is not None:

            mrr += (
                1.0 / rank_found
            )

    recall_1 /= NUM_SAMPLES
    recall_5 /= NUM_SAMPLES
    recall_10 /= NUM_SAMPLES

    mrr /= NUM_SAMPLES

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(
        f"Recall@1  : {recall_1:.4f}"
    )

    print(
        f"Recall@5  : {recall_5:.4f}"
    )

    print(
        f"Recall@10 : {recall_10:.4f}"
    )

    print(
        f"MRR        : {mrr:.4f}"
    )

    print("=" * 60)

    return {
        "method": name,
        "recall@1": recall_1,
        "recall@5": recall_5,
        "recall@10": recall_10,
        "mrr": mrr
    }


print("Loading PubMedQA...")

dataset = load_dataset(
    "qiaojin/PubMedQA",
    "pqa_labeled"
)

print("Loading Retrievers...")

bm25 = BM25Retriever()

dense = DenseRetriever()

hybrid = HybridRetriever()

results = []

results.append(
    evaluate_retriever(
        bm25,
        "BM25"
    )
)

results.append(
    evaluate_retriever(
        dense,
        "Dense"
    )
)

results.append(
    evaluate_retriever(
        hybrid,
        "Hybrid"
    )
)

print("\n")
print("=" * 80)
print("FINAL RESULTS")
print("=" * 80)

for r in results:

    print(
        f"{r['method']:10s} | "
        f"R@1={r['recall@1']:.4f} | "
        f"R@5={r['recall@5']:.4f} | "
        f"R@10={r['recall@10']:.4f} | "
        f"MRR={r['mrr']:.4f}"
    )