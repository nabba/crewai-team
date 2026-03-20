"""Tests for the knowledge base system."""

import json
import os
import tempfile
import pytest

from app.knowledge_base import config
from app.knowledge_base.ingestion import (
    detect_format,
    chunk_text,
    extract_text,
    extract_csv,
    ingest_document,
    DocumentChunk,
    IngestionResult,
)
from app.knowledge_base.vectorstore import KnowledgeStore


# ── Ingestion Tests ──────────────────────────────────────────────────────────


class TestFormatDetection:
    def test_pdf(self):
        assert detect_format("/path/to/file.pdf") == ".pdf"

    def test_docx(self):
        assert detect_format("doc.docx") == ".docx"

    def test_url_http(self):
        assert detect_format("https://example.com/page") == "url"

    def test_url_http_plain(self):
        assert detect_format("http://example.com") == "url"

    def test_txt(self):
        assert detect_format("notes.txt") == ".txt"

    def test_csv(self):
        assert detect_format("data.csv") == ".csv"

    def test_unknown(self):
        assert detect_format("file.xyz") == ".xyz"

    def test_pptx(self):
        assert detect_format("slides.pptx") == ".pptx"


class TestChunking:
    def test_basic_chunking(self):
        text = "Hello world. " * 200  # ~2600 chars
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) > 50

    def test_short_text_filtered(self):
        text = "Too short."
        chunks = chunk_text(text)
        assert chunks == []  # Under 50 char minimum

    def test_preserves_content(self):
        text = "A" * 200 + "\n\n" + "B" * 200
        chunks = chunk_text(text, chunk_size=300, chunk_overlap=50)
        # All content should be captured across chunks
        combined = " ".join(chunks)
        assert "A" * 100 in combined
        assert "B" * 100 in combined


class TestTextExtractor:
    def test_extract_text_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is test content for knowledge base ingestion.\n" * 10)
            f.flush()
            result = extract_text(f.name)
        os.unlink(f.name)
        assert "test content" in result

    def test_extract_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name,value\nAlpha,100\nBeta,200\n")
            f.flush()
            result = extract_csv(f.name)
        os.unlink(f.name)
        assert "Alpha" in result
        assert "100" in result


class TestDocumentChunk:
    def test_chunk_id_deterministic(self):
        c1 = DocumentChunk(text="hello", metadata={"source_path": "/a.txt", "chunk_index": 0})
        c2 = DocumentChunk(text="hello", metadata={"source_path": "/a.txt", "chunk_index": 0})
        assert c1.chunk_id == c2.chunk_id

    def test_chunk_id_differs(self):
        c1 = DocumentChunk(text="hello", metadata={"source_path": "/a.txt", "chunk_index": 0})
        c2 = DocumentChunk(text="hello", metadata={"source_path": "/a.txt", "chunk_index": 1})
        assert c1.chunk_id != c2.chunk_id


class TestIngestion:
    def test_ingest_text_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Knowledge base test document with enough content.\n" * 20)
            f.flush()
            chunks, result = ingest_document(f.name, category="test")
        os.unlink(f.name)
        assert result.success
        assert result.chunks_created > 0
        assert result.format == "txt"

    def test_ingest_missing_file(self):
        chunks, result = ingest_document("/nonexistent/file.pdf")
        assert not result.success
        assert "not found" in result.error.lower() or "No such file" in result.error

    def test_ingest_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")
            f.flush()
            chunks, result = ingest_document(f.name)
        os.unlink(f.name)
        assert not result.success

    def test_metadata_populated(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Enough content for at least one chunk in the knowledge base.\n" * 5)
            f.flush()
            chunks, result = ingest_document(f.name, category="policy", tags=["test", "v1"])
        os.unlink(f.name)
        assert result.success
        assert len(chunks) > 0
        meta = chunks[0].metadata
        assert meta["category"] == "policy"
        assert "test" in json.loads(meta["tags"])
        assert meta["chunk_index"] == 0


# ── Vector Store Tests ───────────────────────────────────────────────────────


class TestKnowledgeStore:
    @pytest.fixture
    def store(self, tmp_path):
        """Create a temporary knowledge store for testing."""
        return KnowledgeStore(
            persist_dir=str(tmp_path / "kb_test"),
            collection_name="test_collection",
        )

    def test_empty_store(self, store):
        stats = store.stats()
        assert stats["total_documents"] == 0
        assert stats["total_chunks"] == 0

    def test_add_and_query_text(self, store):
        result = store.add_text(
            "Our refund policy allows returns within 30 days of purchase. "
            "Customers must provide proof of purchase. Refunds are processed "
            "within 5 business days to the original payment method.",
            source_name="refund_policy",
            category="policy",
        )
        assert result.success
        assert result.chunks_created > 0

        results = store.query("What is the refund policy?")
        assert len(results) > 0
        assert "refund" in results[0]["text"].lower()

    def test_add_document_file(self, store):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(
                "Product specifications for Widget Pro v3.0:\n"
                "Weight: 250g, Dimensions: 10x5x3cm, Battery: 4000mAh.\n"
                "The Widget Pro features a titanium case and gorilla glass display.\n"
                * 5
            )
            f.flush()
            result = store.add_document(f.name, category="product")
        os.unlink(f.name)
        assert result.success

        stats = store.stats()
        assert stats["total_documents"] == 1

    def test_remove_document(self, store):
        result = store.add_text(
            "Temporary knowledge that should be removable from the system. " * 5,
            source_name="temp_doc",
        )
        assert result.success
        assert store.stats()["total_chunks"] > 0

        removed = store.remove_document("manual://temp_doc")
        assert removed > 0
        assert store.stats()["total_chunks"] == 0

    def test_list_documents(self, store):
        store.add_text(
            "Document one content for the knowledge base test suite. " * 5,
            source_name="doc_one",
            category="test",
        )
        store.add_text(
            "Document two content for a different topic in testing. " * 5,
            source_name="doc_two",
            category="other",
        )

        docs = store.list_documents()
        assert len(docs) == 2
        names = {d["source"] for d in docs}
        assert "doc_one" in names
        assert "doc_two" in names

    def test_category_filter(self, store):
        store.add_text(
            "Financial report Q1 2026 showing revenue of 5 million USD. " * 5,
            source_name="finance_report",
            category="finance",
        )
        store.add_text(
            "Technical documentation for the API endpoint configuration. " * 5,
            source_name="api_docs",
            category="technical",
        )

        finance_results = store.query("revenue", category="finance")
        assert len(finance_results) > 0
        assert all(r["category"] == "finance" for r in finance_results)

    def test_reset(self, store):
        store.add_text("Content to be deleted. " * 10, source_name="doomed")
        assert store.stats()["total_chunks"] > 0
        store.reset()
        assert store.stats()["total_chunks"] == 0

    def test_upsert_replaces(self, store):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Original content version one for upsert testing. " * 5)
            f.flush()
            r1 = store.add_document(f.name, category="v1")
        assert r1.success
        count_v1 = store.stats()["total_chunks"]

        # Re-ingest same file — should replace
        with open(f.name, "w") as f2:
            f2.write("Updated content version two for upsert testing. " * 5)
        r2 = store.add_document(f.name, category="v2")
        os.unlink(f.name)

        assert r2.success
        count_v2 = store.stats()["total_chunks"]
        # Should not have doubled the chunks
        assert count_v2 <= count_v1 + 2  # Allow small variance from chunking
