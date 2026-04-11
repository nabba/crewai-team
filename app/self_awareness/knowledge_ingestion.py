"""
knowledge_ingestion.py — AST-based code chunking into ChromaDB self_knowledge.

Ingests the system's own codebase into a searchable ChromaDB collection
so agents can answer detailed questions about their own implementation.

Chunking strategy:
    Python: one chunk per class or function (AST-based)
    Markdown/YAML: section-based splitting
    Incremental: only re-ingests files whose hash has changed

IMMUTABLE — infrastructure-level module.
"""

import ast
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

APP_DIR = Path("/app/app")
WORKSPACE = Path("/app/workspace")
HASH_CACHE_PATH = WORKSPACE / ".self_knowledge_hashes.json"
COLLECTION_NAME = "self_knowledge"

# Files and directories to skip
SKIP_DIRS = {"__pycache__", ".git", "node_modules", ".venv", "venv"}
CODE_EXTENSIONS = {".py", ".yaml", ".yml", ".md"}


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _load_hash_cache() -> dict:
    if HASH_CACHE_PATH.exists():
        try:
            return json.loads(HASH_CACHE_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_hash_cache(cache: dict) -> None:
    from app.safe_io import safe_write_json
    safe_write_json(HASH_CACHE_PATH, cache)


# ── Python AST chunking ──────────────────────────────────────────────────────


def _chunk_python(source: str, filepath: str) -> list[dict]:
    """Split Python source into chunks: one per class/function.

    Falls back to whole-file if AST parsing fails.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [{"text": source[:4000], "type": "file", "name": filepath, "source_file": filepath}]

    chunks = []
    lines = source.splitlines()

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, "end_lineno") and node.end_lineno else start + 20
            chunk_text = "\n".join(lines[start:end])

            node_type = "class" if isinstance(node, ast.ClassDef) else "function"
            chunks.append({
                "text": chunk_text[:4000],
                "type": node_type,
                "name": f"{filepath}::{node.name}",
                "source_file": filepath,
                "node_name": node.name,
            })

    # If no classes/functions found, chunk the whole file
    if not chunks:
        chunks.append({
            "text": source[:4000],
            "type": "module",
            "name": filepath,
            "source_file": filepath,
        })

    return chunks


# ── Markdown/YAML chunking ───────────────────────────────────────────────────


def _chunk_markdown(content: str, filepath: str) -> list[dict]:
    """Split markdown by headers."""
    chunks = []
    current_section = ""
    current_text = []

    for line in content.splitlines():
        if line.startswith("#"):
            if current_text:
                chunks.append({
                    "text": "\n".join(current_text)[:4000],
                    "type": "section",
                    "name": f"{filepath}::{current_section}" if current_section else filepath,
                    "source_file": filepath,
                })
            current_section = line.lstrip("#").strip()
            current_text = [line]
        else:
            current_text.append(line)

    if current_text:
        chunks.append({
            "text": "\n".join(current_text)[:4000],
            "type": "section",
            "name": f"{filepath}::{current_section}" if current_section else filepath,
            "source_file": filepath,
        })

    return chunks


def _chunk_yaml(content: str, filepath: str) -> list[dict]:
    """Treat YAML as a single chunk (usually config files)."""
    return [{"text": content[:4000], "type": "config", "name": filepath, "source_file": filepath}]


# ── Ingestion pipeline ────────────────────────────────────────────────────────


def ingest_codebase(full: bool = False) -> dict:
    """Ingest the system's codebase into ChromaDB self_knowledge collection.

    Args:
        full: If True, re-ingest everything. If False, only changed files.

    Returns: {files_processed, chunks_added, files_skipped}
    """
    try:
        from app.memory.chromadb_manager import get_client
        client = get_client()
        collection = client.get_or_create_collection(
            COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as e:
        return {"error": f"ChromaDB unavailable: {e}"}

    hash_cache = {} if full else _load_hash_cache()
    new_cache = {}
    stats = {"files_processed": 0, "chunks_added": 0, "files_skipped": 0}

    # Scan app directory
    scan_dirs = [APP_DIR]
    # Also scan soul files and workspace configs
    souls = APP_DIR / "souls"
    if souls.exists():
        scan_dirs.append(souls)

    all_files = []
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
        for f in scan_dir.rglob("*"):
            if f.is_file() and f.suffix in CODE_EXTENSIONS:
                if not any(skip in f.parts for skip in SKIP_DIRS):
                    all_files.append(f)

    # Also add key workspace files
    for wf in [WORKSPACE / "system_chronicle.md"]:
        if wf.exists():
            all_files.append(wf)

    for filepath in all_files:
        try:
            rel_path = str(filepath.relative_to(APP_DIR.parent))
            current_hash = _file_hash(filepath)
            new_cache[rel_path] = current_hash

            # Skip unchanged files (incremental)
            if not full and hash_cache.get(rel_path) == current_hash:
                stats["files_skipped"] += 1
                continue

            content = filepath.read_text(encoding="utf-8", errors="ignore")
            if not content.strip():
                continue

            # Chunk based on file type
            if filepath.suffix == ".py":
                chunks = _chunk_python(content, rel_path)
            elif filepath.suffix == ".md":
                chunks = _chunk_markdown(content, rel_path)
            elif filepath.suffix in (".yaml", ".yml"):
                chunks = _chunk_yaml(content, rel_path)
            else:
                continue

            # Upsert into ChromaDB
            if chunks:
                ids = [hashlib.sha256(c["name"].encode()).hexdigest()[:16] for c in chunks]
                documents = [c["text"] for c in chunks]
                metadatas = [{
                    "source_file": c["source_file"],
                    "type": c["type"],
                    "name": c["name"],
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                } for c in chunks]

                collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
                stats["chunks_added"] += len(chunks)

            stats["files_processed"] += 1

        except Exception as e:
            logger.debug(f"knowledge_ingestion: failed to ingest {filepath}: {e}")

    _save_hash_cache(new_cache)
    logger.info(f"knowledge_ingestion: {stats}")
    return stats


def query_self_knowledge(query: str, n_results: int = 5) -> list[dict]:
    """Search the self_knowledge collection."""
    try:
        from app.memory.chromadb_manager import get_client
        client = get_client()
        collection = client.get_or_create_collection(COLLECTION_NAME)
        if collection.count() == 0:
            return []
        results = collection.query(query_texts=[query], n_results=n_results)
        output = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            output.append({"document": doc, "metadata": meta})
        return output
    except Exception:
        return []
