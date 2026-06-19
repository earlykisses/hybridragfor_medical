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

retriever = HybridRetriever()

results = retriever.retrieve(
    "Do mitochondria play a role in programmed cell death?",
    top_k=3
)

for item in results:
    print("\n")
    print("ID:", item[0])
    print("Score:", item[1])
    print("Text:", item[2][:200])