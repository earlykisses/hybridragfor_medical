import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(
    path="vector_db"
)

collection = client.get_collection(
    "faithfulmed_pubmedqa"
)

model = SentenceTransformer(
    "pritamdeka/S-PubMedBert-MS-MARCO"
)

query = "Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?"

query_embedding = model.encode(query)

results = collection.query(
    query_embeddings=[
        query_embedding.tolist()
    ],
    n_results=5
)

for i, doc in enumerate(
    results["documents"][0]
):
    print("\n")
    print("="*80)
    print(f"RESULT {i+1}")
    print(doc[:1000])
    print(results["distances"][0])