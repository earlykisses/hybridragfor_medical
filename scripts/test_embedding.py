from sentence_transformers import SentenceTransformer

print("Loading model...")

model = SentenceTransformer(
    "pritamdeka/S-PubMedBert-MS-MARCO"
)

text = "SGLT2 inhibitors reduce cardiovascular mortality."

embedding = model.encode(text)

print("Embedding shape:")
print(embedding.shape)