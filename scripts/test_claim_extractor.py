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

from generation.claim_extractor import ClaimExtractor

extractor = ClaimExtractor()

explanation = """
The provided evidence demonstrates that mitochondrial
dynamics are involved in developmentally regulated
programmed cell death in lace plants.

Treatment with cyclosporine A reduced perforation
formation.

DNA fragmentation correlated with mitochondrial stages.
"""

claims = extractor.extract_claims(
    explanation
)

print("\n")
print("=" * 60)
print("EXTRACTED CLAIMS")
print("=" * 60)

for i, claim in enumerate(
    claims,
    start=1
):
    print(f"Claim {i}: {claim}")

print("=" * 60)