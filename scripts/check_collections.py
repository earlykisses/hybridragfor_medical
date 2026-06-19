import chromadb

client = chromadb.PersistentClient(
    path="vector_db"
)

collections = client.list_collections()

print("\nCollections Found:\n")

for c in collections:
    print(c.name)