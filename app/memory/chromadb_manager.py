import chromadb
import threading
import uuid
from sentence_transformers import SentenceTransformer
from pathlib import Path

PERSIST_DIR = Path("/app/workspace/memory")
TEAM_COLLECTION = "team_shared"

_model = SentenceTransformer("all-MiniLM-L6-v2")  # Runs locally, no API

# Thread-safe singleton — prevents lock contention when multiple threads
# each try to create their own PersistentClient pointing to the same dir.
_client = None
_client_lock = threading.Lock()


def get_client():
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            _client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    return _client


def store(collection_name: str, text: str, metadata: dict = None):
    client = get_client()
    col = client.get_or_create_collection(collection_name)
    embedding = _model.encode(text).tolist()
    col.add(
        documents=[text],
        embeddings=[embedding],
        metadatas=[metadata or {}],
        ids=[str(uuid.uuid4())],
    )


def retrieve(collection_name: str, query: str, n: int = 5) -> list[str]:
    client = get_client()
    col = client.get_or_create_collection(collection_name)
    if col.count() == 0:
        return []
    embedding = _model.encode(query).tolist()
    results = col.query(
        query_embeddings=[embedding], n_results=min(n, col.count())
    )
    return results["documents"][0]


def store_team(text: str, metadata: dict = None):
    """Store in the shared team-wide collection (cross-crew sharing)."""
    store(TEAM_COLLECTION, text, metadata)


def retrieve_team(query: str, n: int = 5) -> list[str]:
    """Retrieve from the shared team-wide collection."""
    return retrieve(TEAM_COLLECTION, query, n)
