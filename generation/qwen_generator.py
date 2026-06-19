from ollama import chat


class QwenGenerator:

    def generate(
        self,
        question,
        evidence
    ):

        prompt = f"""
You are an evidence-grounded medical assistant.

Use ONLY the provided evidence.

Do NOT use outside medical knowledge.

If the evidence is insufficient, return:

INSUFFICIENT EVIDENCE

Question:
{question}

Evidence:
{evidence}

Return EXACTLY in this format:

ANSWER:
(short direct answer)

EXPLANATION:
(step-by-step reasoning based only on the evidence)
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

        return response["message"]["content"]