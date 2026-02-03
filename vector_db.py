from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class QdrantStorage:
    def __init__(
        self,
        path="qdrant_storage",
        collection="docs",
        dim=768,
    ):
        self.client = QdrantClient(path=path, timeout=30)
        self.collection = collection

        if not self.client.collection_exists(collection_name=self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE,
                ),
            )

    def upsert(self, ids, vectors, payloads):
        points = [
            PointStruct(
                id=ids[i],
                vector=vectors[i],
                payload=payloads[i],
            )
            for i in range(len(ids))
        ]

        self.client.upsert(
            collection_name=self.collection,
            points=points,
        )

    def search(self, query_vector, top_k: int = 5):
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points

        contexts = []
        sources = set()

        for point in response:
            payload = point.payload or {}
            text = payload.get("text")
            source = payload.get("source")

            if text:
                contexts.append(text)
            if source:
                sources.add(source)

        return {
            "contexts": contexts,
            "sources": list(sources),
        }
