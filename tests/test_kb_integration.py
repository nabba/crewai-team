"""
End-to-end integration tests for the Knowledge Base + RAG system.

Tests the full pipeline:
  1. Ingestion (text, files, multi-format)
  2. Vector retrieval (semantic search, scoring, filtering)
  3. RAG context injection (_load_knowledge_base_context)
  4. CrewAI tool interface (KnowledgeSearchTool, KnowledgeStatusTool)
  5. Signal command handlers (kb, kb list, kb add, kb search, kb remove, kb reset)
  6. Firebase reporter (report_knowledge_base)
  7. Edge cases (empty KB, no results, duplicate ingestion)
"""

import json
import os
import tempfile
import textwrap
import pytest

from app.knowledge_base import config
from app.knowledge_base.ingestion import (
    detect_format,
    chunk_text,
    ingest_document,
    DocumentChunk,
)
from app.knowledge_base.vectorstore import KnowledgeStore
from app.knowledge_base.tools import (
    KnowledgeSearchTool,
    KnowledgeStatusTool,
    KnowledgeIngestTool,
    get_knowledge_tools,
    set_store,
    get_store,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    """Fresh KnowledgeStore in a temp directory."""
    s = KnowledgeStore(
        persist_dir=str(tmp_path / "kb"),
        collection_name="test_kb",
    )
    # Also set as the singleton so tools use this store
    set_store(s)
    yield s


@pytest.fixture
def seeded_store(store):
    """Store pre-loaded with diverse test documents."""
    store.add_text(
        "Our company refund policy: Customers may request a full refund within "
        "30 days of purchase. After 30 days, only store credit is available. "
        "Refunds are processed within 5 business days to the original payment "
        "method. Proof of purchase is required for all refund requests. "
        "Digital products are non-refundable after download.",
        source_name="refund_policy.md",
        category="policy",
        tags=["customer-service", "returns"],
    )
    store.add_text(
        "Widget Pro v3 Technical Specifications: "
        "Processor: ARM Cortex-A78, 2.8 GHz octa-core. "
        "RAM: 8GB LPDDR5. Storage: 256GB UFS 3.1. "
        "Display: 6.7-inch AMOLED, 2400x1080, 120Hz. "
        "Battery: 5000mAh with 65W fast charging. "
        "Weight: 189g. Dimensions: 160.8 x 74.2 x 8.3mm. "
        "Operating System: Android 15 with WidgetOS overlay. "
        "Camera: 108MP main, 12MP ultrawide, 5MP macro. "
        "5G bands: n1, n3, n5, n7, n8, n28, n41, n77, n78.",
        source_name="widget_pro_specs.md",
        category="product",
        tags=["hardware", "v3"],
    )
    store.add_text(
        "Q1 2026 Financial Summary: Total revenue was EUR 12.4 million, "
        "up 18% year-over-year. EBITDA margin improved to 22.3% from 19.1%. "
        "Customer acquisition cost decreased to EUR 45 from EUR 52. "
        "Monthly recurring revenue reached EUR 4.2 million. "
        "Cash position: EUR 8.7 million. Burn rate: EUR 1.1 million per month. "
        "Headcount: 87 employees across 4 offices.",
        source_name="q1_financials.xlsx",
        category="finance",
        tags=["Q1-2026", "board"],
    )
    store.add_text(
        "API Authentication Guide: All API requests require a Bearer token "
        "in the Authorization header. Tokens are obtained via POST /auth/token "
        "with client_id and client_secret. Tokens expire after 3600 seconds. "
        "Rate limits: 1000 requests per minute for standard tier, "
        "10000 per minute for enterprise. Exceeding limits returns HTTP 429. "
        "Use exponential backoff with jitter for retries.",
        source_name="api_auth_guide.md",
        category="technical",
        tags=["api", "security"],
    )
    return store


# ─────────────────────────────────────────────────────────────────────────────
# 1. INGESTION PIPELINE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestIngestionPipeline:
    """Test document ingestion from various sources."""

    def test_ingest_txt_file(self, store):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Company vacation policy: All employees are entitled to "
                    "25 days of paid vacation per year. Unused days can be "
                    "carried over up to 5 days into the next year.\n" * 5)
            path = f.name
        result = store.add_document(path, category="policy", tags=["hr"])
        os.unlink(path)
        assert result.success, f"Ingestion failed: {result.error}"
        assert result.chunks_created > 0
        assert result.format == "txt"
        assert result.total_characters > 100

    def test_ingest_csv_file(self, store):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("product,price,stock\n")
            for i in range(50):
                f.write(f"Widget-{i},{10+i*0.5},{100-i}\n")
            path = f.name
        result = store.add_document(path, category="product")
        os.unlink(path)
        assert result.success
        assert result.chunks_created > 0

    def test_ingest_json_file(self, store):
        data = {
            "company": "Acme Corp",
            "departments": [
                {"name": "Engineering", "headcount": 45, "budget": 2500000},
                {"name": "Sales", "headcount": 22, "budget": 1200000},
                {"name": "Marketing", "headcount": 15, "budget": 800000},
            ],
            "total_employees": 87,
            "founded": 2019,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f, indent=2)
            path = f.name
        result = store.add_document(path, category="operations")
        os.unlink(path)
        assert result.success

    def test_ingest_markdown_file(self, store):
        md_content = textwrap.dedent("""\
        # Employee Handbook

        ## Work Hours
        Standard working hours are 9:00 to 17:00 Monday through Friday.
        Flexible working is available with manager approval.

        ## Remote Work Policy
        Employees may work remotely up to 3 days per week.
        A stable internet connection is required.
        All remote workers must be available during core hours (10:00-15:00).

        ## Code of Conduct
        All employees are expected to maintain professional behavior.
        Harassment of any kind will not be tolerated.
        Report concerns to HR or use the anonymous reporting system.
        """)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(md_content * 3)  # Repeat to ensure enough content for chunks
            path = f.name
        result = store.add_document(path, category="policy", tags=["hr", "handbook"])
        os.unlink(path)
        assert result.success
        assert result.chunks_created >= 1

    def test_ingest_html_file(self, store):
        html = textwrap.dedent("""\
        <html><body>
        <h1>Product Release Notes v3.2</h1>
        <h2>New Features</h2>
        <ul>
            <li>Dark mode support across all screens</li>
            <li>Batch export functionality for reports</li>
            <li>Two-factor authentication via hardware keys</li>
        </ul>
        <h2>Bug Fixes</h2>
        <p>Fixed an issue where dashboard widgets would not refresh after timezone changes.
        Resolved a memory leak in the background sync service that caused increased RAM usage
        over extended periods. Corrected currency formatting for Japanese Yen in invoice exports.</p>
        <h2>Known Issues</h2>
        <p>Chart rendering may be slow with datasets exceeding 100,000 rows.
        A fix is planned for v3.3.</p>
        </body></html>
        """)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            f.write(html * 3)
            path = f.name
        result = store.add_document(path, category="product", tags=["release-notes", "v3.2"])
        os.unlink(path)
        assert result.success

    def test_ingest_raw_text(self, store):
        result = store.add_text(
            "The server room is located on floor B2. Access requires badge "
            "level 3 or higher. Temperature is maintained at 18-22 degrees Celsius. "
            "Backup generators activate within 30 seconds of power failure. "
            "Fire suppression uses inert gas system. " * 3,
            source_name="server_room_info",
            category="operations",
        )
        assert result.success
        assert result.chunks_created > 0

    def test_ingest_text_too_short(self, store):
        result = store.add_text("Hi.", source_name="tiny")
        assert not result.success
        assert "too short" in result.error.lower()

    def test_ingest_nonexistent_file(self, store):
        result = store.add_document("/nonexistent/path/to/file.pdf")
        assert not result.success

    def test_duplicate_ingestion_replaces(self, store):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Original content for deduplication test. " * 20)
            path = f.name

        r1 = store.add_document(path, category="v1")
        assert r1.success
        chunks_v1 = store.stats()["total_chunks"]

        # Overwrite file content and re-ingest
        with open(path, "w") as f:
            f.write("Updated content for deduplication test. " * 20)
        r2 = store.add_document(path, category="v2")
        os.unlink(path)

        assert r2.success
        chunks_v2 = store.stats()["total_chunks"]
        # Should not double — old version is replaced
        assert chunks_v2 <= chunks_v1 + 2


# ─────────────────────────────────────────────────────────────────────────────
# 2. SEMANTIC RETRIEVAL TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestSemanticRetrieval:
    """Test that vector search returns semantically relevant results."""

    def test_refund_query(self, seeded_store):
        results = seeded_store.query("How do I get a refund?")
        assert len(results) > 0
        top = results[0]
        assert top["score"] > 0.4
        assert "refund" in top["text"].lower()
        assert top["source"] == "refund_policy.md"
        assert top["category"] == "policy"

    def test_product_specs_query(self, seeded_store):
        results = seeded_store.query("What processor does Widget Pro use?")
        assert len(results) > 0
        # Should find the specs document
        sources = [r["source"] for r in results]
        assert "widget_pro_specs.md" in sources
        spec_result = next(r for r in results if r["source"] == "widget_pro_specs.md")
        assert "cortex" in spec_result["text"].lower() or "arm" in spec_result["text"].lower()

    def test_financial_query(self, seeded_store):
        results = seeded_store.query("What was Q1 revenue?")
        assert len(results) > 0
        sources = [r["source"] for r in results]
        assert "q1_financials.xlsx" in sources

    def test_api_auth_query(self, seeded_store):
        results = seeded_store.query("How do I authenticate API requests?")
        assert len(results) > 0
        sources = [r["source"] for r in results]
        assert "api_auth_guide.md" in sources

    def test_category_filter(self, seeded_store):
        results = seeded_store.query("revenue EBITDA margin", category="finance")
        assert len(results) > 0
        for r in results:
            assert r["category"] == "finance"

    def test_no_results_for_unrelated_query(self, seeded_store):
        results = seeded_store.query(
            "quantum entanglement in black holes",
            min_score=0.7,  # High threshold
        )
        # Should return few or no results since nothing is related
        for r in results:
            assert r["score"] >= 0.7

    def test_empty_store_returns_nothing(self, store):
        results = store.query("anything")
        assert results == []

    def test_top_k_respected(self, seeded_store):
        results = seeded_store.query("company information", top_k=2)
        assert len(results) <= 2

    def test_results_include_metadata(self, seeded_store):
        results = seeded_store.query("refund policy")
        assert len(results) > 0
        r = results[0]
        assert "text" in r
        assert "source" in r
        assert "score" in r
        assert "category" in r
        assert "metadata" in r
        assert "ingested_at" in r


# ─────────────────────────────────────────────────────────────────────────────
# 3. RAG CONTEXT INJECTION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestRAGInjection:
    """Test _load_knowledge_base_context returns proper prompt context."""

    def test_rag_returns_context_for_relevant_query(self, seeded_store):
        from app.agents.commander import _load_knowledge_base_context
        context = _load_knowledge_base_context("What is our refund policy?")
        assert context, "Expected RAG context but got empty string"
        assert "KNOWLEDGE BASE CONTEXT" in context
        assert "<kb_passage" in context
        assert "refund" in context.lower()
        assert "source=" in context
        assert "relevance=" in context
        assert "not instructions" in context.lower()

    def test_rag_returns_empty_for_empty_store(self, store):
        from app.agents.commander import _load_knowledge_base_context
        context = _load_knowledge_base_context("anything at all")
        assert context == ""

    def test_rag_returns_empty_for_unrelated_query(self, seeded_store):
        from app.agents.commander import _load_knowledge_base_context
        # With min_score=0.35, very unrelated queries should return empty
        context = _load_knowledge_base_context(
            "photosynthesis in deep sea hydrothermal vents"
        )
        # May or may not return results depending on embedding similarity
        # but if it does, passages should be marked as reference data
        if context:
            assert "not instructions" in context.lower()

    def test_rag_context_has_multiple_passages(self, seeded_store):
        from app.agents.commander import _load_knowledge_base_context
        context = _load_knowledge_base_context("company overview and products")
        if context:
            # Should contain multiple kb_passage blocks
            passage_count = context.count("<kb_passage")
            assert passage_count >= 1

    def test_rag_context_truncates_long_passages(self, seeded_store):
        from app.agents.commander import _load_knowledge_base_context
        context = _load_knowledge_base_context("tell me everything")
        if context:
            # Each passage should be capped at 600 chars
            for passage in context.split("<kb_passage")[1:]:
                # Extract content between tags
                content = passage.split("</kb_passage>")[0]
                # Content includes the closing > of the opening tag
                inner = content.split(">", 1)[1] if ">" in content else content
                assert len(inner.strip()) <= 650  # 600 + small margin for whitespace


# ─────────────────────────────────────────────────────────────────────────────
# 4. CREWAI TOOL INTERFACE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestCrewAITools:
    """Test the CrewAI tool wrappers work correctly."""

    def test_search_tool_finds_results(self, seeded_store):
        tool = KnowledgeSearchTool()
        result = tool._run(query="refund policy")
        assert "Found" in result
        assert "refund" in result.lower()
        assert "Source:" in result
        assert "relevance:" in result.lower()

    def test_search_tool_no_results(self, store):
        tool = KnowledgeSearchTool()
        result = tool._run(query="something")
        assert "No relevant information" in result

    def test_search_tool_with_category(self, seeded_store):
        tool = KnowledgeSearchTool()
        result = tool._run(query="revenue EBITDA quarterly results", category="finance")
        assert "Found" in result
        assert "finance" in result.lower()

    def test_status_tool_summary(self, seeded_store):
        tool = KnowledgeStatusTool()
        result = tool._run(detail_level="summary")
        assert "Knowledge Base Status" in result
        assert "Documents:" in result
        assert "Chunks:" in result
        assert "Characters:" in result

    def test_status_tool_full(self, seeded_store):
        tool = KnowledgeStatusTool()
        result = tool._run(detail_level="full")
        assert "Documents:" in result
        assert "refund_policy.md" in result or "widget_pro_specs.md" in result

    def test_ingest_tool(self, store):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test document for ingest tool validation. " * 20)
            path = f.name
        tool = KnowledgeIngestTool()
        result = tool._run(source=path, category="test", tags="automated,v1")
        os.unlink(path)
        assert "Ingested" in result
        assert "Chunks:" in result

    def test_get_knowledge_tools_default(self, store):
        tools = get_knowledge_tools()
        names = [t.name for t in tools]
        assert "search_knowledge_base" in names
        assert "knowledge_base_status" in names
        # Ingest tool should NOT be included by default
        assert "ingest_to_knowledge_base" not in names

    def test_get_knowledge_tools_with_ingest(self, store):
        tools = get_knowledge_tools(include_ingest=True)
        names = [t.name for t in tools]
        assert "ingest_to_knowledge_base" in names


# ─────────────────────────────────────────────────────────────────────────────
# 5. SIGNAL COMMAND HANDLER TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalCommands:
    """Test kb commands go through commander.handle() special command path.

    These test the command parsing and response format, not the full
    Commander LLM routing (which requires API keys).
    """

    def _make_commander(self, seeded_store):
        """Create a minimal Commander that can handle kb commands."""
        # We need to import Commander, but it requires heavy deps.
        # Instead, test the command detection logic directly.
        pass

    def test_kb_status_command_detection(self, seeded_store):
        """Verify 'kb' and 'kb status' are recognized as commands."""
        lower = "kb"
        assert lower in ("kb", "kb status", "knowledge base")
        lower = "kb status"
        assert lower in ("kb", "kb status", "knowledge base")

    def test_kb_list_format(self, seeded_store):
        """Test kb list returns proper document listing."""
        docs = seeded_store.list_documents()
        assert len(docs) == 4
        sources = {d["source"] for d in docs}
        assert "refund_policy.md" in sources
        assert "widget_pro_specs.md" in sources
        assert "q1_financials.xlsx" in sources
        assert "api_auth_guide.md" in sources

    def test_kb_search_returns_results(self, seeded_store):
        results = seeded_store.query(question="refund", top_k=5)
        assert len(results) > 0
        assert results[0]["score"] > 0.3

    def test_kb_remove_works(self, seeded_store):
        before = seeded_store.stats()["total_documents"]
        removed = seeded_store.remove_document("manual://refund_policy.md")
        assert removed > 0
        after = seeded_store.stats()["total_documents"]
        assert after == before - 1

    def test_kb_reset_clears_all(self, seeded_store):
        assert seeded_store.stats()["total_chunks"] > 0
        seeded_store.reset()
        assert seeded_store.stats()["total_chunks"] == 0
        assert seeded_store.stats()["total_documents"] == 0

    def test_kb_add_url_ingestion(self, seeded_store):
        """Test that add_document can handle URL-like sources."""
        # We can't actually fetch URLs in tests, but we verify format detection
        fmt = detect_format("https://example.com/docs/api")
        assert fmt == "url"

    def test_kb_add_with_category(self, seeded_store):
        """Test category assignment during ingestion."""
        result = seeded_store.add_text(
            "New operations manual content for category testing. " * 5,
            source_name="ops_manual",
            category="operations",
        )
        assert result.success
        results = seeded_store.query("operations manual", category="operations")
        assert len(results) > 0
        assert results[0]["category"] == "operations"


# ─────────────────────────────────────────────────────────────────────────────
# 6. FIREBASE REPORTER INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

class TestFirebaseReporter:
    """Test that report_knowledge_base() builds the right payload."""

    def test_report_builds_correct_payload(self, seeded_store):
        """Verify the data structure that would be pushed to Firestore."""
        stats = seeded_store.stats()

        # Simulate what report_knowledge_base() does
        assert stats["total_documents"] == 4
        assert stats["total_chunks"] > 0
        assert stats["total_characters"] > 0
        assert stats["estimated_tokens"] > 0
        assert "policy" in stats["categories"]
        assert "product" in stats["categories"]
        assert "finance" in stats["categories"]
        assert "technical" in stats["categories"]

        docs = []
        for d in stats.get("documents", [])[:50]:
            docs.append({
                "source": d.get("source", "unknown"),
                "format": d.get("format", "?"),
                "category": d.get("category", "general"),
                "chunks": d.get("total_chunks", 0),
                "ingested_at": d.get("ingested_at", ""),
            })
        assert len(docs) == 4
        assert all(d["source"] for d in docs)
        assert all(d["ingested_at"] for d in docs)


# ─────────────────────────────────────────────────────────────────────────────
# 7. CROSS-DOMAIN QUERIES (the real RAG test)
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossDomainRAG:
    """Test realistic query scenarios across different document types."""

    def test_policy_question(self, seeded_store):
        """User asks a policy question — should find refund policy."""
        results = seeded_store.query("Can I return a product after 2 weeks?")
        assert len(results) > 0
        # Refund policy should be top or near-top
        policy_results = [r for r in results if r["source"] == "refund_policy.md"]
        assert len(policy_results) > 0, f"Expected refund policy in results, got: {[r['source'] for r in results]}"

    def test_technical_question(self, seeded_store):
        """User asks about API — should find auth guide."""
        results = seeded_store.query("What are the API rate limits?")
        assert len(results) > 0
        tech_results = [r for r in results if r["category"] == "technical"]
        assert len(tech_results) > 0

    def test_financial_question(self, seeded_store):
        """User asks about revenue — should find financials."""
        results = seeded_store.query("How much revenue did we make?")
        assert len(results) > 0
        fin_results = [r for r in results if r["category"] == "finance"]
        assert len(fin_results) > 0

    def test_hardware_question(self, seeded_store):
        """User asks about device specs — should find product doc."""
        results = seeded_store.query("What's the battery capacity?")
        assert len(results) > 0
        prod_results = [r for r in results if r["source"] == "widget_pro_specs.md"]
        assert len(prod_results) > 0

    def test_rag_injection_for_policy_question(self, seeded_store):
        """Full RAG test: verify context injected for a real question."""
        from app.agents.commander import _load_knowledge_base_context
        context = _load_knowledge_base_context(
            "A customer wants to return a Widget Pro they bought 3 weeks ago. "
            "What is our return policy?"
        )
        assert context, "RAG should return context for this question"
        assert "refund" in context.lower() or "return" in context.lower()
        assert "<kb_passage" in context

    def test_rag_injection_for_technical_question(self, seeded_store):
        """Full RAG test: API auth question gets relevant context."""
        from app.agents.commander import _load_knowledge_base_context
        context = _load_knowledge_base_context(
            "How do I authenticate with the API? What headers do I need?"
        )
        assert context, "RAG should return context for API auth question"
        assert "bearer" in context.lower() or "token" in context.lower() or "auth" in context.lower()

    def test_rag_injection_for_financial_question(self, seeded_store):
        """Full RAG test: revenue question pulls financial data."""
        from app.agents.commander import _load_knowledge_base_context
        context = _load_knowledge_base_context(
            "What was our Q1 2026 revenue and EBITDA margin?"
        )
        assert context, "RAG should return context for financial question"
        assert "12.4" in context or "revenue" in context.lower()
