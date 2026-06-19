import pickle
import chromadb

from sentence_transformers import SentenceTransformer


class HybridRetriever:

    def __init__(self):

        print("Loading Chroma...")

        self.client = chromadb.PersistentClient(
            path="vector_db"
        )

        self.collection = self.client.get_collection(
            "faithfulmed_corpus"
        )

        print("Loading BM25...")

        with open(
            "retrieval/bm25_full.pkl",
            "rb"
        ) as f:

            data = pickle.load(f)

        self.bm25 = data["bm25"]
        self.documents = data["documents"]
        self.ids = data["ids"]

        print("Building ID Lookup...")

        self.id_to_doc = {
            doc_id: doc
            for doc_id, doc in zip(
                self.ids,
                self.documents
            )
        }

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            "pritamdeka/S-PubMedBert-MS-MARCO",
            device="cuda"
        )

    def dense_search(
        self,
        query,
        k=20
    ):

        embedding = self.model.encode(
            query
        )

        results = self.collection.query(
            query_embeddings=[
                embedding.tolist()
            ],
            n_results=k
        )

        return results["ids"][0]

    def bm25_search(
        self,
        query,
        k=20
    ):

        scores = self.bm25.get_scores(
            query.lower().split()
        )

        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        return [
            self.ids[idx]
            for idx in ranked
        ]

    def reciprocal_rank_fusion(
        self,
        dense_ids,
        bm25_ids,
        k=60
    ):

        scores = {}

        for rank, doc_id in enumerate(
            dense_ids
        ):

            scores[doc_id] = (
                scores.get(doc_id, 0)
                + 1 / (k + rank + 1)
            )

        for rank, doc_id in enumerate(
            bm25_ids
        ):

            scores[doc_id] = (
                scores.get(doc_id, 0)
                + 1 / (k + rank + 1)
            )

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked

    def retrieve(
        self,
        query,
        top_k=10
    ):

        dense_ids = self.dense_search(
            query,
            k=20
        )

        bm25_ids = self.bm25_search(
            query,
            k=20
        )

        fused = self.reciprocal_rank_fusion(
            dense_ids,
            bm25_ids
        )

        results = []

        for doc_id, score in fused[:top_k]:

            text = self.id_to_doc.get(
                doc_id,
                ""
            )

            results.append(
                (
                    doc_id,
                    score,
                    text
                )
            )

        return results