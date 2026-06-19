import pickle

with open(
    "retrieval/bm25.pkl",
    "rb"
) as f:

    data = pickle.load(f)

bm25 = data["bm25"]
documents = data["documents"]

query = "mitochondria programmed cell death"

tokenized_query = query.lower().split()

scores = bm25.get_scores(
    tokenized_query
)

top_indices = sorted(
    range(len(scores)),
    key=lambda i: scores[i],
    reverse=True
)[:5]

for rank, idx in enumerate(top_indices):

    print("\n")
    print("="*80)
    print(f"BM25 RESULT {rank+1}")
    print(documents[idx][:1000])