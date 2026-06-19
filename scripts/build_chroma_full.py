import json
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BATCH_SIZE = 512

print("Loading model...")

model = SentenceTransformer(
    "pritamdeka/S-PubMedBert-MS-MARCO",
    device="cuda"
)

print("Creating Chroma DB...")

client = chromadb.PersistentClient(
    path="vector_db"
)

collection = client.get_or_create_collection(
    "faithfulmed_full"
)

documents = []
ids = []

with open(
    "data/merged_corpus.jsonl",
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        record = json.loads(line)

        ids.append(record["id"])
        documents.append(record["text"])

print(f"Loaded {len(documents)} passages")

for start in tqdm(
    range(0, len(documents), BATCH_SIZE)
):

    end = start + BATCH_SIZE

    batch_docs = documents[start:end]
    batch_ids = ids[start:end]

    embeddings = model.encode(
        batch_docs,
        batch_size=32,
        show_progress_bar=False
    )

    collection.add(
        ids=batch_ids,
        documents=batch_docs,
        embeddings=embeddings.tolist()
    )

print("\nDone.")