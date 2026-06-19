from ollama import chat


class ClaimExtractor:

    def extract_claims(
        self,
        explanation
    ):

        prompt = f"""
You are an expert medical fact extraction system.

Extract atomic factual claims from the explanation.

Rules:

1. One claim per line.
2. Keep claims short.
3. Remove opinions.
4. Remove speculation.
5. Only include verifiable medical statements.
6. Do not number claims.

EXPLANATION:

{explanation}

CLAIMS:
"""

        response = chat(
            model="qwen2.5:7b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        claims_text = response["message"]["content"]

        claims = [
            line.strip()
            for line in claims_text.split("\n")
            if line.strip()
        ]

        return claims