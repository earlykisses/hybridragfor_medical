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

from retrieval.dense_retriever import DenseRetriever

retriever = DenseRetriever()

query = "Does aspirin reduce cardiovascular risk?"

results = retriever.retrieve(
    query,
    top_k=5
)

for i, result in enumerate(results):

    doc_id, text, score = result

    print("\n")
    print("=" * 80)
    print(f"RESULT {i+1}")
    print(f"ID: {doc_id}")
    print(f"Distance: {score:.4f}")
    print(text[:1000])