import pickle


class BM25Retriever:

    def __init__(self):

        print("Loading BM25 Index...")

        with open(
            "retrieval/bm25_full.pkl",
            "rb"
        ) as f:

            data = pickle.load(f)

        self.bm25 = data["bm25"]
        self.documents = data["documents"]
        self.ids = data["ids"]

    def retrieve(
        self,
        query,
        top_k=10
    ):

        scores = self.bm25.get_scores(
            query.lower().split()
        )

        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []

        for idx in ranked:

            results.append(
                (
                    self.ids[idx],
                    self.documents[idx],
                    float(scores[idx])
                )
            )

        return results