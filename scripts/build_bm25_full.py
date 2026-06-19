import json
import pickle
from rank_bm25 import BM25Okapi
from tqdm import tqdm

print("Loading merged corpus...")

documents = []
ids = []

with open(
    "data/merged_corpus.jsonl",
    "r",
    encoding="utf-8"
) as f:

    for line in tqdm(f):

        record = json.loads(line)

        documents.append(
            record["text"]
        )

        ids.append(
            record["id"]
        )

print(f"\nLoaded {len(documents)} passages")

print("\nTokenizing corpus...")

tokenized_corpus = [
    doc.lower().split()
    for doc in tqdm(documents)
]

print("\nBuilding BM25 index...")

bm25 = BM25Okapi(
    tokenized_corpus
)

output = {
    "bm25": bm25,
    "documents": documents,
    "ids": ids
}

with open(
    "retrieval/bm25_full.pkl",
    "wb"
) as f:

    pickle.dump(
        output,
        f
    )

print("\nBM25 index saved.")

print(
    "Output: retrieval/bm25_full.pkl"
)