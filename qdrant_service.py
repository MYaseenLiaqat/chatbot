import hashlib
import os
import uuid
from typing import Any

from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

load_dotenv("backend/.env")
load_dotenv(".env")

COLLECTION_NAME = "chatbot_documents"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
VECTOR_DIMENSION = 384
DISTANCE = models.Distance.COSINE
DEFAULT_LIMIT = 4

def qdrant_log(message: str) -> None:
    """Write Qdrant diagnostics to the same terminal as the Chainlit app."""

    print(message, flush=True)


class QdrantService:
    """Synchronous Qdrant/FastEmbed operations used through worker threads."""

    def __init__(self) -> None:
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.client: QdrantClient | None = None
        self.embedding_model: TextEmbedding | None = None

    def initialize(self) -> None:
        """Connect to Qdrant, load embeddings, and verify the existing collection."""

        if not self.url or not self.api_key:
            raise RuntimeError(
                "QDRANT_URL and QDRANT_API_KEY must be configured."
            )

        qdrant_log("[QDRANT] Connecting to configured cloud endpoint.")
        client = QdrantClient(url=self.url, api_key=self.api_key, timeout=30)
        try:
            if not client.collection_exists(COLLECTION_NAME):
                raise RuntimeError(
                    f"Qdrant collection '{COLLECTION_NAME}' does not exist. "
                    "Create it with the existing setup script first."
                )

            info = client.get_collection(COLLECTION_NAME)
            vectors = info.config.params.vectors
            vector_size = getattr(vectors, "size", None)
            distance = getattr(vectors, "distance", None)
            if vector_size != VECTOR_DIMENSION or distance != DISTANCE:
                raise RuntimeError(
                    f"Collection '{COLLECTION_NAME}' must use "
                    f"{VECTOR_DIMENSION}-dimensional cosine vectors."
                )
            for field_name in ("session_id", "document_id"):
                client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )
                qdrant_log(f"[QDRANT] Payload index verified: {field_name}.")

            qdrant_log(
                f"[QDRANT] Collection verified: {COLLECTION_NAME} "
                f"({vector_size} dimensions, {distance})."
            )
            qdrant_log(f"[QDRANT] Loading embedding model: {EMBEDDING_MODEL}")
            embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
            next(embedding_model.embed(["dimension check"]))
        except Exception:
            client.close()
            raise

        self.client = client
        self.embedding_model = embedding_model
        qdrant_log("[QDRANT] Connection and embedding model ready.")

    def _require_ready(self) -> tuple[QdrantClient, TextEmbedding]:
        if self.client is None or self.embedding_model is None:
            raise RuntimeError("Qdrant service has not been initialized.")
        return self.client, self.embedding_model

    def verify_connection(self) -> dict[str, Any]:
        """Return safe connection and collection details."""

        client, _ = self._require_ready()
        info = client.get_collection(COLLECTION_NAME)
        return {
            "collection": COLLECTION_NAME,
            "vector_dimension": getattr(info.config.params.vectors, "size", None),
            "distance": str(getattr(info.config.params.vectors, "distance", "")),
        }

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate and validate 384-dimensional vectors for document or query text."""

        _, embedding_model = self._require_ready()
        vectors = [vector.tolist() for vector in embedding_model.embed(texts)]
        if len(vectors) != len(texts):
            raise RuntimeError("Embedding model returned an unexpected vector count.")
        if any(len(vector) != VECTOR_DIMENSION for vector in vectors):
            raise RuntimeError(
                f"Embedding dimension mismatch; expected {VECTOR_DIMENSION}."
            )
        return vectors

    @staticmethod
    def document_id(session_id: str, file_bytes: bytes) -> str:
        """Create a stable per-session document ID from the file contents."""

        digest = hashlib.sha256(file_bytes).hexdigest()
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{session_id}:{digest}"))

    @staticmethod
    def point_id(document_id: str, chunk_index: int) -> str:
        """Create a deterministic point ID for duplicate-safe upserts."""

        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{chunk_index}"))

    def upsert_document(
        self,
        chunks: list[dict],
        session_id: str,
        document_id: str,
    ) -> int:
        """Embed and batch-upsert document chunks, replacing duplicate uploads."""

        client, _ = self._require_ready()
        vectors = self.embed_texts([chunk["text"] for chunk in chunks])
        self.delete_document(document_id, session_id)
        points = []
        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            point_id = self.point_id(document_id, index)
            payload = {
                "text": chunk["text"],
                "filename": chunk["filename"],
                "page": chunk["page"],
                "chunk_index": index,
                "document_id": document_id,
                "session_id": session_id,
                "chunk_id": chunk["id"],
            }
            points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))

        batch_size = 64
        for start in range(0, len(points), batch_size):
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points[start:start + batch_size],
                wait=True,
            )
            qdrant_log(
                f"[QDRANT] Inserted vectors {start + 1}-"
                f"{min(start + batch_size, len(points))} of {len(points)}."
            )
        return len(points)

    def retrieve(
        self,
        question: str,
        session_id: str,
        limit: int = DEFAULT_LIMIT,
    ) -> list[dict]:
        """Retrieve semantically similar chunks restricted to one Chainlit session."""

        client, _ = self._require_ready()
        query_vector = self.embed_texts([question])[0]
        result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=session_id),
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
        )
        retrieved = []
        for point in result.points:
            payload = point.payload or {}
            retrieved.append({
                "id": str(point.id),
                "score": float(point.score),
                "text": payload.get("text", ""),
                "filename": payload.get("filename", ""),
                "page": payload.get("page"),
                "chunk_index": payload.get("chunk_index"),
                "document_id": payload.get("document_id"),
                "session_id": payload.get("session_id"),
                "chunk_id": payload.get("chunk_id"),
            })
        return retrieved

    def delete_document(self, document_id: str, session_id: str) -> None:
        """Delete one document only when it belongs to the requesting session."""

        client, _ = self._require_ready()
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        ),
                        models.FieldCondition(
                            key="session_id",
                            match=models.MatchValue(value=session_id),
                        ),
                    ]
                )
            ),
            wait=True,
        )

    def delete_session(self, session_id: str) -> None:
        """Delete all vectors for one session without touching other sessions."""

        client, _ = self._require_ready()
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="session_id",
                            match=models.MatchValue(value=session_id),
                        )
                    ]
                )
            ),
            wait=True,
        )


qdrant_service = QdrantService()
