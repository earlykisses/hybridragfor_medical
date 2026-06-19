import json
import pickle
from rank_bm25 import BM25Okapi
from tqdm import tqdm

documents = []
doc_ids = []
tokenized_docs = []

with open(
    "data/corpus.jsonl",
    "r",
    encoding="utf-8"
) as f:

    for line in tqdm(f):

        item = json.loads(line)

        doc_ids.append(item["id"])
        documents.append(item["text"])

        tokenized_docs.append(
            item["text"].lower().split()
        )

bm25 = BM25Okapi(tokenized_docs)

with open(
    "retrieval/bm25.pkl",
    "wb"
) as f:

    pickle.dump(
        {
            "bm25": bm25,
            "documents": documents,
            "ids": doc_ids
        },
        f
    )

print("BM25 index saved")