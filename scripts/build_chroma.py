import json
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer

print("Loading embedding model...")

model = SentenceTransformer(
    "pritamdeka/S-PubMedBert-MS-MARCO"
)

print("Loading ChromaDB...")

client = chromadb.PersistentClient(
    path="vector_db"
)

collection = client.get_or_create_collection(
    name="faithfulmed_pubmedqa"
)

print("Loading corpus...")

docs = []
ids = []

with open(
    "data/corpus.jsonl",
    "r",
    encoding="utf-8"
) as f:

    for line in f:
        item = json.loads(line)

        ids.append(item["id"])
        docs.append(item["text"])

print(f"Loaded {len(docs)} documents")

BATCH_SIZE = 100

for i in tqdm(
    range(0, len(docs), BATCH_SIZE)
):

    batch_docs = docs[i:i+BATCH_SIZE]
    batch_ids = ids[i:i+BATCH_SIZE]

    embeddings = model.encode(
        batch_docs,
        show_progress_bar=False
    )

    collection.add(
        ids=batch_ids,
        documents=batch_docs,
        embeddings=embeddings.tolist()
    )

print("\nIndexing complete.")