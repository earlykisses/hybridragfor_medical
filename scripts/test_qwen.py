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

from generation.qwen_generator import QwenGenerator

generator = QwenGenerator()

question = (
    "Does aspirin reduce cardiovascular risk?"
)

evidence = """
Aspirin is widely used for secondary prevention
of cardiovascular disease.

Aspirin reduces the risk of myocardial infarction,
stroke, and cardiovascular death.
"""

response = generator.generate(
    question,
    evidence
)

print("\n")
print("=" * 80)
print(response)
print("=" * 80)