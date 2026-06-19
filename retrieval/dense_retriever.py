import chromadb

from sentence_transformers import SentenceTransformer


class DenseRetriever:

    def __init__(self):

        print("Loading Chroma...")

        self.client = chromadb.PersistentClient(
            path="vector_db"
        )

        self.collection = self.client.get_collection(
            "faithfulmed_corpus"
        )

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            "pritamdeka/S-PubMedBert-MS-MARCO",
            device="cuda"
        )

    def retrieve(
        self,
        query,
        top_k=10
    ):

        embedding = self.model.encode(
            query
        )

        results = self.collection.query(
            query_embeddings=[
                embedding.tolist()
            ],
            n_results=top_k
        )

        retrieved = []

        for i in range(
            len(results["ids"][0])
        ):

            retrieved.append(
                (
                    results["ids"][0][i],
                    results["documents"][0][i],
                    float(
                        results["distances"][0][i]
                    )
                )
            )

        return retrieved