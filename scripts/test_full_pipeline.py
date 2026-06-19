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
from generation.claim_extractor import ClaimExtractor


def main():

    print("=" * 80)
    print("FAITHFULMED FULL PIPELINE")
    print("=" * 80)

    retriever = HybridRetriever()

    generator = QwenGenerator()

    extractor = ClaimExtractor()

    question = input(
        "\nEnter Medical Question:\n> "
    )

    print("\nRetrieving Evidence...")

    retrieved_docs = retriever.retrieve(
        question,
        top_k=10
    )

    evidence = ""

    for idx, doc in enumerate(
        retrieved_docs,
        start=1
    ):

        evidence += (
            f"Evidence {idx}:\n"
            f"{doc[2]}\n\n"
        )

    print(
        "\nGenerating Answer & Explanation..."
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

    print("\n")
    print("=" * 80)
    print("EXTRACTING CLAIMS")
    print("=" * 80)

    claims = extractor.extract_claims(
        response
    )

    for idx, claim in enumerate(
        claims,
        start=1
    ):

        print(
            f"Claim {idx}: {claim}"
        )

    print("\n")
    print("=" * 80)
    print("RETRIEVED EVIDENCE")
    print("=" * 80)

    for idx, doc in enumerate(
        retrieved_docs,
        start=1
    ):

        print(f"\nEvidence {idx}")

        print(
            f"ID: {doc[0]}"
        )

        print(
            f"Score: {doc[1]:.4f}"
        )

        print(
            doc[2][:500]
        )

        print(
            "-" * 80
        )


if __name__ == "__main__":
    main()