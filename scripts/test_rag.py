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
from generation.qwen_generator import QwenGenerator


retriever = HybridRetriever()

generator = QwenGenerator()

question = (
    "Do mitochondria play a role in "
    "programmed cell death?"
)

results = retriever.retrieve(
    question,
    top_k=10
)

evidence = ""

for idx, doc in enumerate(
    results,
    start=1
):

    evidence += (
        f"Evidence {idx}:\n"
        f"{doc[2]}\n\n"
    )

response = generator.generate(
    question,
    evidence
)

print("\n")
print("=" * 80)
print("RAG OUTPUT")
print("=" * 80)

print(response)

print("=" * 80)