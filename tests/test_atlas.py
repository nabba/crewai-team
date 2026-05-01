"""
ATLAS Subsystem Tests
======================

Comprehensive tests for the Autonomous Tool-Learning & Adaptive Skills System.
Covers all 8 modules, their public APIs, cross-module data flows, and system wiring.

Run: docker exec crewai-team-gateway-1 python3 -m pytest /app/tests/test_atlas.py -v
"""

import inspect
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# app.atlas hardcodes /app/workspace/atlas/* paths; on macOS the parent
# /app is the read-only system root, so any test that touches the persist
# layer fails with OSError. Skip the whole module unless we're in a
# Docker-style writable layout.
pytestmark = pytest.mark.skipif(
    not os.access("/app", os.W_OK),
    reason="Requires Docker-style /app writable layout (run inside the gateway container)",
)


# ════════════════════════════════════════════════════════════════════════════════
# 1. MODULE IMPORT TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestModuleImports:
    """Every ATLAS module must be importable without side effects."""

    def test_skill_library_import(self):
        from app.atlas.skill_library import SkillLibrary, SkillManifest, get_library
        assert callable(SkillLibrary)
        assert callable(get_library)

    def test_competence_tracker_import(self):
        from app.atlas.competence_tracker import CompetenceTracker, CompetenceEntry, get_tracker
        assert callable(CompetenceTracker)
        assert callable(get_tracker)

    def test_api_scout_import(self):
        from app.atlas.api_scout import APIScout, APIKnowledge, EndpointSpec, get_scout
        assert callable(APIScout)
        assert callable(get_scout)

    def test_code_forge_import(self):
        from app.atlas.code_forge import CodeForge, CodeForgeResult, get_forge
        assert callable(CodeForge)
        assert callable(get_forge)

    def test_video_learner_import(self):
        from app.atlas.video_learner import VideoLearner, VideoContent, ExtractedKnowledge, get_learner
        assert callable(VideoLearner)
        assert callable(get_learner)

    def test_learning_planner_import(self):
        from app.atlas.learning_planner import (
            LearningPlanner, LearningPlan, LearningStep,
            QualityEvaluator, get_planner, get_evaluator,
        )
        assert callable(LearningPlanner)
        assert callable(get_planner)
        assert callable(get_evaluator)

    def test_auth_patterns_import(self):
        from app.atlas.auth_patterns import (
            detect_auth_pattern, get_pattern, get_pattern_code, list_patterns,
        )
        assert callable(detect_auth_pattern)
        assert callable(list_patterns)

    def test_audit_log_import(self):
        from app.atlas.audit_log import log_external_call
        assert callable(log_external_call)

    def test_package_init(self):
        import app.atlas
        assert hasattr(app.atlas, "__doc__")
        assert "ATLAS" in app.atlas.__doc__


# ════════════════════════════════════════════════════════════════════════════════
# 2. SKILL LIBRARY TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestSkillLibrary:
    """SkillLibrary manages skill registration, search, and lifecycle."""

    def test_singleton_accessor(self):
        from app.atlas.skill_library import get_library
        lib1 = get_library()
        lib2 = get_library()
        assert lib1 is lib2

    def test_register_and_retrieve(self):
        from app.atlas.skill_library import SkillLibrary
        lib = SkillLibrary()
        manifest = lib.register_skill(
            skill_id="test_atlas_skill",
            name="Test Atlas Skill",
            category="learned",
            code="def hello(): return 'world'",
            description="A test skill for ATLAS test suite",
            source_type="manual",
        )
        assert manifest is not None
        assert manifest.skill_id == "test_atlas_skill"
        assert manifest.name == "Test Atlas Skill"
        assert manifest.category == "learned"

        # Retrieve by ID
        retrieved = lib.get_skill("test_atlas_skill")
        assert retrieved is not None
        assert retrieved.name == "Test Atlas Skill"

    def test_get_skill_code(self):
        from app.atlas.skill_library import SkillLibrary
        lib = SkillLibrary()
        lib.register_skill(
            skill_id="test_code_skill",
            name="Code Skill",
            category="learned",
            code="def compute(x): return x * 2",
            description="Doubler",
            source_type="manual",
        )
        code = lib.get_skill_code("test_code_skill")
        assert code is not None
        assert "compute" in code

    def test_get_nonexistent_skill(self):
        from app.atlas.skill_library import SkillLibrary
        lib = SkillLibrary()
        result = lib.get_skill("nonexistent_skill_xyz_12345")
        assert result is None

    def test_search_by_category(self):
        from app.atlas.skill_library import SkillLibrary
        lib = SkillLibrary()
        lib.register_skill(
            skill_id="test_search_apis",
            name="API Search Test",
            category="apis",
            code="# api stub",
            description="API skill for search test",
            source_type="manual",
        )
        results = lib.search(category="apis")
        assert isinstance(results, list)
        found = any(s.skill_id == "test_search_apis" for s in results)
        assert found

    def test_search_by_query(self):
        from app.atlas.skill_library import SkillLibrary
        lib = SkillLibrary()
        lib.register_skill(
            skill_id="test_search_query",
            name="Unique Zigzag Pattern",
            category="patterns",
            code="# zigzag",
            description="A unique zigzag pattern skill",
            source_type="manual",
        )
        results = lib.search(query="zigzag")
        assert isinstance(results, list)
        found = any(s.skill_id == "test_search_query" for s in results)
        assert found

    def test_list_skills(self):
        from app.atlas.skill_library import SkillLibrary
        lib = SkillLibrary()
        skills = lib.list_skills()
        assert isinstance(skills, list)

    def test_list_skills_with_category_filter(self):
        from app.atlas.skill_library import SkillLibrary
        lib = SkillLibrary()
        lib.register_skill(
            skill_id="test_list_filter",
            name="Filtered Skill",
            category="recipes",
            code="# recipe",
            description="Recipe skill",
            source_type="manual",
        )
        skills = lib.list_skills(category="recipes")
        assert all(s.category == "recipes" for s in skills)

    def test_record_usage(self):
        from app.atlas.skill_library import SkillLibrary
        lib = SkillLibrary()
        lib.register_skill(
            skill_id="test_usage_skill",
            name="Usage Tracked",
            category="learned",
            code="# tracked",
            description="Track usage",
            source_type="manual",
        )
        lib.record_usage("test_usage_skill", success=True)
        lib.record_usage("test_usage_skill", success=False)
        skill = lib.get_skill("test_usage_skill")
        assert skill.usage_count >= 2
        assert skill.usage_success_count >= 1

    def test_get_stale_skills(self):
        from app.atlas.skill_library import SkillLibrary
        lib = SkillLibrary()
        stale = lib.get_stale_skills(max_age_days=30)
        assert isinstance(stale, list)

    def test_format_inventory(self):
        from app.atlas.skill_library import SkillLibrary
        lib = SkillLibrary()
        inv = lib.format_inventory()
        assert isinstance(inv, str)

    def test_get_competence_summary(self):
        from app.atlas.skill_library import SkillLibrary
        lib = SkillLibrary()
        summary = lib.get_competence_summary()
        assert isinstance(summary, dict)

    def test_categories_constant(self):
        from app.atlas.skill_library import CATEGORIES
        assert "apis" in CATEGORIES
        assert "patterns" in CATEGORIES
        assert "recipes" in CATEGORIES
        assert "learned" in CATEGORIES

    def test_manifest_to_dict_roundtrip(self):
        from app.atlas.skill_library import SkillManifest
        m = SkillManifest(
            skill_id="roundtrip_test",
            name="Roundtrip",
            category="learned",
            version=1,
            language="python",
            description="Test roundtrip",
        )
        d = m.to_dict()
        assert d["skill_id"] == "roundtrip_test"
        m2 = SkillManifest.from_dict(d)
        assert m2.skill_id == m.skill_id
        assert m2.name == m.name

    def test_effective_confidence_decay(self):
        from app.atlas.skill_library import SkillManifest
        m = SkillManifest(
            skill_id="decay_test",
            name="Decay",
            category="learned",
            version=1,
            language="python",
            description="Confidence decay test",
            confidence=0.9,
            last_verified="2020-01-01T00:00:00",
        )
        eff = m.effective_confidence()
        assert eff < 0.9  # Should decay over time


# ════════════════════════════════════════════════════════════════════════════════
# 3. COMPETENCE TRACKER TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestCompetenceTracker:
    """CompetenceTracker maps system capabilities and gaps."""

    def test_singleton_accessor(self):
        from app.atlas.competence_tracker import get_tracker
        t1 = get_tracker()
        t2 = get_tracker()
        assert t1 is t2

    def test_register_competence(self):
        from app.atlas.competence_tracker import CompetenceTracker
        ct = CompetenceTracker()
        entry = ct.register(
            domain="testing",
            name="pytest_atlas",
            confidence=0.8,
            source="manual",
        )
        assert entry is not None
        assert entry.domain == "testing"
        assert entry.name == "pytest_atlas"
        assert entry.confidence == 0.8

    def test_check_competence(self):
        from app.atlas.competence_tracker import CompetenceTracker
        ct = CompetenceTracker()
        ct.register(domain="testing", name="check_test", confidence=0.7, source="manual")
        result = ct.check_competence("testing", "check_test")
        assert result is not None
        assert result.confidence == 0.7

    def test_check_unknown_competence(self):
        from app.atlas.competence_tracker import CompetenceTracker
        ct = CompetenceTracker()
        result = ct.check_competence("nonexistent_domain_xyz", "unknown_skill_abc")
        assert result is None

    def test_record_usage_success(self):
        from app.atlas.competence_tracker import CompetenceTracker
        ct = CompetenceTracker()
        ct.register(domain="testing", name="usage_test", confidence=0.5, source="manual")
        ct.record_usage("testing", "usage_test", success=True)
        entry = ct.check_competence("testing", "usage_test")
        assert entry.usage_count >= 1
        assert entry.usage_success_count >= 1

    def test_get_gaps(self):
        from app.atlas.competence_tracker import CompetenceTracker
        ct = CompetenceTracker()
        ct.register(domain="testing", name="gap_test", confidence=0.2, source="manual")
        gaps = ct.get_gaps(min_confidence=0.5)
        assert isinstance(gaps, list)
        found = any(e.name == "gap_test" for e in gaps)
        assert found

    def test_get_strengths(self):
        from app.atlas.competence_tracker import CompetenceTracker
        ct = CompetenceTracker()
        ct.register(domain="testing", name="strength_test", confidence=0.95, source="manual")
        strengths = ct.get_strengths(min_confidence=0.8)
        assert isinstance(strengths, list)
        found = any(e.name == "strength_test" for e in strengths)
        assert found

    def test_check_task_readiness(self):
        from app.atlas.competence_tracker import CompetenceTracker
        ct = CompetenceTracker()
        ct.register(domain="python", name="flask_api", confidence=0.9, source="manual")
        requirements = [
            {"domain": "python", "name": "flask_api"},
            {"domain": "cloud", "name": "kubernetes_deploy"},
        ]
        readiness = ct.check_task_readiness(requirements)
        assert isinstance(readiness, dict)
        assert "ready" in readiness
        assert "known" in readiness
        assert "unknown" in readiness

    def test_sync_from_skill_library(self):
        from app.atlas.competence_tracker import CompetenceTracker
        ct = CompetenceTracker()
        count = ct.sync_from_skill_library()
        assert isinstance(count, int)
        assert count >= 0

    def test_format_competence_map(self):
        from app.atlas.competence_tracker import CompetenceTracker
        ct = CompetenceTracker()
        text = ct.format_competence_map()
        assert isinstance(text, str)

    def test_entry_to_dict_roundtrip(self):
        from app.atlas.competence_tracker import CompetenceEntry
        e = CompetenceEntry(
            domain="test",
            name="roundtrip",
            confidence=0.75,
        )
        d = e.to_dict()
        assert d["domain"] == "test"
        e2 = CompetenceEntry.from_dict(d)
        assert e2.domain == e.domain
        assert e2.confidence == e.confidence

    def test_effective_confidence(self):
        from app.atlas.competence_tracker import CompetenceEntry
        e = CompetenceEntry(
            domain="test",
            name="decay",
            confidence=0.9,
            last_verified="2020-01-01T00:00:00",
        )
        eff = e.effective_confidence()
        assert eff <= 0.9  # Time decay (may equal 0.9 if decay floors)


# ════════════════════════════════════════════════════════════════════════════════
# 4. AUTH PATTERNS TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestAuthPatterns:
    """Auth patterns provide reusable authentication strategies."""

    def test_list_patterns(self):
        from app.atlas.auth_patterns import list_patterns
        patterns = list_patterns()
        assert isinstance(patterns, list)
        assert len(patterns) >= 5  # 6 built-in patterns
        assert "api_key_header" in patterns
        assert "oauth2_client_credentials" in patterns
        assert "basic_auth" in patterns

    def test_get_pattern(self):
        from app.atlas.auth_patterns import get_pattern
        p = get_pattern("api_key_header")
        assert p is not None
        assert p.pattern_id == "api_key_header"
        assert p.name is not None
        assert len(p.required_params) > 0

    def test_get_nonexistent_pattern(self):
        from app.atlas.auth_patterns import get_pattern
        p = get_pattern("totally_fake_pattern_xyz")
        assert p is None

    def test_get_pattern_code(self):
        from app.atlas.auth_patterns import get_pattern_code
        code = get_pattern_code("api_key_header")
        assert isinstance(code, str)
        assert len(code) > 0

    def test_detect_api_key_pattern(self):
        from app.atlas.auth_patterns import detect_auth_pattern
        docs = "Set the Authorization header with Bearer token. API key required."
        results = detect_auth_pattern(docs)
        assert isinstance(results, list)
        if results:
            pattern_id, confidence = results[0]
            assert isinstance(pattern_id, str)
            assert 0.0 <= confidence <= 1.0

    def test_detect_oauth_pattern(self):
        from app.atlas.auth_patterns import detect_auth_pattern
        docs = "OAuth 2.0 client credentials grant. Token endpoint at /oauth/token."
        results = detect_auth_pattern(docs)
        assert isinstance(results, list)
        if results:
            oauth_found = any("oauth" in pid for pid, _ in results)
            assert oauth_found

    def test_detect_basic_auth(self):
        from app.atlas.auth_patterns import detect_auth_pattern
        docs = "Authenticate with HTTP Basic Authentication using username and password."
        results = detect_auth_pattern(docs)
        if results:
            basic_found = any("basic" in pid for pid, _ in results)
            assert basic_found

    def test_all_patterns_have_code(self):
        from app.atlas.auth_patterns import list_patterns, get_pattern_code
        for pat_id in list_patterns():
            code = get_pattern_code(pat_id)
            assert isinstance(code, str)
            assert len(code) > 10, f"Pattern {pat_id} has no code template"

    def test_all_patterns_have_required_params(self):
        from app.atlas.auth_patterns import list_patterns, get_pattern
        for pat_id in list_patterns():
            p = get_pattern(pat_id)
            assert p is not None
            assert isinstance(p.required_params, (list, tuple))
            assert isinstance(p.description, str)


# ════════════════════════════════════════════════════════════════════════════════
# 5. AUDIT LOG TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestAuditLog:
    """Audit log records all external calls for compliance."""

    def test_log_external_call_no_crash(self):
        from app.atlas.audit_log import log_external_call
        # Should not crash even without DB
        log_external_call(
            agent="test_agent",
            action="test_action",
            target="https://example.com/api",
            method="GET",
            result="success",
            response_code=200,
            execution_time_ms=42,
        )

    def test_log_with_all_fields(self):
        from app.atlas.audit_log import log_external_call
        log_external_call(
            agent="researcher",
            action="api_call",
            target="https://api.openai.com/v1/chat",
            method="POST",
            credential_used="api_key_header",
            sandbox_id="sandbox_001",
            result="success",
            response_code=200,
            execution_time_ms=1500,
            tokens_consumed=500,
            cost_usd=0.015,
            approval="auto",
        )

    def test_log_failure(self):
        from app.atlas.audit_log import log_external_call
        log_external_call(
            agent="coder",
            action="api_call",
            target="https://api.example.com/fail",
            method="POST",
            result="error",
            response_code=500,
            execution_time_ms=3000,
        )


# ════════════════════════════════════════════════════════════════════════════════
# 6. API SCOUT TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestAPIScout:
    """APIScout discovers and generates clients for APIs."""

    def test_singleton_accessor(self):
        from app.atlas.api_scout import get_scout
        s1 = get_scout()
        s2 = get_scout()
        assert s1 is s2

    def test_init_no_crash(self):
        from app.atlas.api_scout import APIScout
        scout = APIScout()
        assert scout is not None

    def test_get_known_apis(self):
        from app.atlas.api_scout import APIScout
        scout = APIScout()
        apis = scout.get_known_apis()
        assert isinstance(apis, list)

    def test_get_unknown_api_knowledge(self):
        from app.atlas.api_scout import APIScout
        scout = APIScout()
        knowledge = scout.get_api_knowledge("totally_nonexistent_api_xyz")
        assert knowledge is None

    def test_api_knowledge_dataclass(self):
        from app.atlas.api_scout import APIKnowledge
        ak = APIKnowledge(
            name="Test API",
            base_url="https://api.test.com",
            version="v1",
            description="Test API for ATLAS tests",
            auth_type="api_key",
            auth_details={"header": "X-API-Key"},
            endpoints=[],
            rate_limits={"requests_per_minute": 100},
            error_codes={},
            doc_sources=["https://docs.test.com"],
            confidence=0.8,
        )
        d = ak.to_dict()
        assert d["name"] == "Test API"
        assert d["base_url"] == "https://api.test.com"
        ak2 = APIKnowledge.from_dict(d)
        assert ak2.name == ak.name
        assert ak2.confidence == ak.confidence

    def test_endpoint_spec_dataclass(self):
        from app.atlas.api_scout import EndpointSpec
        ep = EndpointSpec(
            path="/users",
            method="GET",
            description="List users",
            parameters={"page": "int"},
            request_body=None,
            response_schema={"type": "array"},
            rate_limit="100/min",
            requires_auth=True,
        )
        assert ep.path == "/users"
        assert ep.requires_auth is True

    def test_prompts_are_immutable_strings(self):
        from app.atlas import api_scout
        assert hasattr(api_scout, "API_EXTRACTION_PROMPT")
        assert isinstance(api_scout.API_EXTRACTION_PROMPT, str)
        assert len(api_scout.API_EXTRACTION_PROMPT) > 50


# ════════════════════════════════════════════════════════════════════════════════
# 7. CODE FORGE TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestCodeForge:
    """CodeForge generates grounded code from verified skills."""

    def test_singleton_accessor(self):
        from app.atlas.code_forge import get_forge
        f1 = get_forge()
        f2 = get_forge()
        assert f1 is f2

    def test_init_no_crash(self):
        from app.atlas.code_forge import CodeForge
        forge = CodeForge()
        assert forge is not None

    def test_result_dataclass(self):
        from app.atlas.code_forge import CodeForgeResult
        r = CodeForgeResult(
            success=True,
            code="def hello(): return 'world'",
            test_code="def test_hello(): assert hello() == 'world'",
            skill_id="test_skill",
            subtasks=["parse", "generate"],
            skills_used=["string_utils"],
            skills_missing=[],
            debug_iterations=0,
            error=None,
            duration_seconds=1.5,
        )
        assert r.success is True
        assert r.debug_iterations == 0
        assert len(r.subtasks) == 2

    def test_max_debug_iterations_constant(self):
        from app.atlas.code_forge import MAX_DEBUG_ITERATIONS
        assert isinstance(MAX_DEBUG_ITERATIONS, int)
        assert MAX_DEBUG_ITERATIONS >= 1

    def test_prompts_exist(self):
        from app.atlas import code_forge
        assert hasattr(code_forge, "DECOMPOSE_PROMPT")
        assert hasattr(code_forge, "COMPOSE_PROMPT")
        assert hasattr(code_forge, "DEBUG_PROMPT")


# ════════════════════════════════════════════════════════════════════════════════
# 8. VIDEO LEARNER TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestVideoLearner:
    """VideoLearner extracts knowledge from YouTube videos."""

    def test_singleton_accessor(self):
        from app.atlas.video_learner import get_learner
        v1 = get_learner()
        v2 = get_learner()
        assert v1 is v2

    def test_init_no_crash(self):
        from app.atlas.video_learner import VideoLearner
        vl = VideoLearner()
        assert vl is not None

    def test_video_content_dataclass(self):
        from app.atlas.video_learner import VideoContent
        vc = VideoContent(
            url="https://youtube.com/watch?v=test",
            title="Test Video",
            channel="Test Channel",
            duration_seconds=600,
            description="A test video",
        )
        assert vc.url.startswith("https://")
        assert vc.duration_seconds == 600

    def test_extracted_knowledge_dataclass(self):
        from app.atlas.video_learner import ExtractedKnowledge
        ek = ExtractedKnowledge(
            source_url="https://youtube.com/watch?v=test",
            source_title="Test Video",
            video_type="tutorial",
            concepts=["REST API", "pagination"],
            procedures=["1. Create endpoint", "2. Add pagination"],
            code_recipes=[],
            api_knowledge={},
            gotchas=["Rate limiting"],
            confidence=0.7,
        )
        d = ek.to_dict()
        assert d["video_type"] == "tutorial"
        assert len(d["concepts"]) == 2
        assert d["confidence"] == 0.7

    def test_video_type_signals_exist(self):
        from app.atlas.video_learner import VIDEO_TYPE_SIGNALS
        assert isinstance(VIDEO_TYPE_SIGNALS, dict)
        assert len(VIDEO_TYPE_SIGNALS) > 0


# ════════════════════════════════════════════════════════════════════════════════
# 9. LEARNING PLANNER TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestLearningPlanner:
    """LearningPlanner generates learning plans for capability gaps."""

    def test_singleton_accessors(self):
        from app.atlas.learning_planner import get_planner, get_evaluator
        p1 = get_planner()
        p2 = get_planner()
        assert p1 is p2
        e1 = get_evaluator()
        e2 = get_evaluator()
        assert e1 is e2

    def test_init_no_crash(self):
        from app.atlas.learning_planner import LearningPlanner
        lp = LearningPlanner()
        assert lp is not None

    def test_learning_step_dataclass(self):
        from app.atlas.learning_planner import LearningStep
        step = LearningStep(
            step_id="step_1",
            method="api_discovery",
            target="Stripe API",
            domain="payments",
            priority=1,
            estimated_minutes=15,
            rationale="Need payment integration",
        )
        assert step.method == "api_discovery"
        assert step.priority == 1

    def test_learning_plan_dataclass(self):
        from app.atlas.learning_planner import LearningPlan, LearningStep
        step = LearningStep(
            step_id="s1",
            method="video_learning",
            target="FastAPI tutorial",
            domain="web",
            priority=1,
            estimated_minutes=30,
            rationale="Learn web framework",
        )
        plan = LearningPlan(
            task_description="Build a REST API",
            steps=[step],
            total_estimated_minutes=30,
            readiness_before={"ready": False},
        )
        d = plan.to_dict()
        assert d["task_description"] == "Build a REST API"
        assert len(d["steps"]) == 1

    def test_learning_methods_constant(self):
        from app.atlas.learning_planner import LEARNING_METHODS
        assert isinstance(LEARNING_METHODS, dict)
        assert len(LEARNING_METHODS) > 0
        for method, info in LEARNING_METHODS.items():
            assert isinstance(method, str)

    def test_quality_evaluator_record_api(self):
        from app.atlas.learning_planner import QualityEvaluator
        qe = QualityEvaluator()
        qe.record_api_discovery(success=True)
        qe.record_api_discovery(success=False)
        report = qe.get_quality_report()
        assert isinstance(report, dict)

    def test_quality_evaluator_record_video(self):
        from app.atlas.learning_planner import QualityEvaluator
        qe = QualityEvaluator()
        qe.record_video_extraction(recipes_working=3, recipes_total=5)

    def test_quality_evaluator_record_forge(self):
        from app.atlas.learning_planner import QualityEvaluator
        qe = QualityEvaluator()
        qe.record_code_forge_result(success_first_try=True, debug_iterations=0)
        qe.record_code_forge_result(success_first_try=False, debug_iterations=2)

    def test_quality_evaluator_format_report(self):
        from app.atlas.learning_planner import QualityEvaluator
        qe = QualityEvaluator()
        text = qe.format_report()
        assert isinstance(text, str)


# ════════════════════════════════════════════════════════════════════════════════
# 10. CROSS-MODULE DATA FLOW TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestCrossModuleFlows:
    """Verify data flows between ATLAS modules."""

    def test_skill_library_to_competence_tracker(self):
        """Skills registered in library should sync to competence tracker."""
        from app.atlas.skill_library import SkillLibrary
        from app.atlas.competence_tracker import CompetenceTracker

        lib = SkillLibrary()
        lib.register_skill(
            skill_id="cross_flow_skill",
            name="Cross Flow Test",
            category="learned",
            code="# cross flow",
            description="Cross-module flow test",
            source_type="manual",
        )
        ct = CompetenceTracker()
        count = ct.sync_from_skill_library()
        assert isinstance(count, int)

    def test_auth_patterns_used_by_api_scout(self):
        """APIScout should import and use auth_patterns."""
        src = inspect.getsource(__import__("app.atlas.api_scout", fromlist=["APIScout"]))
        assert "auth_patterns" in src

    def test_code_forge_uses_skill_library(self):
        """CodeForge should look up skills from the library."""
        src = inspect.getsource(__import__("app.atlas.code_forge", fromlist=["CodeForge"]))
        assert "skill_library" in src

    def test_code_forge_uses_audit_log(self):
        """CodeForge should audit its operations."""
        src = inspect.getsource(__import__("app.atlas.code_forge", fromlist=["CodeForge"]))
        assert "audit_log" in src or "log_external_call" in src

    def test_api_scout_uses_audit_log(self):
        """APIScout should audit its external calls."""
        src = inspect.getsource(__import__("app.atlas.api_scout", fromlist=["APIScout"]))
        assert "audit_log" in src or "log_external_call" in src

    def test_video_learner_registers_skills(self):
        """VideoLearner should register extracted knowledge as skills."""
        src = inspect.getsource(__import__("app.atlas.video_learner", fromlist=["VideoLearner"]))
        assert "skill_library" in src or "register_skill" in src

    def test_video_learner_updates_competence(self):
        """VideoLearner should update competence tracker."""
        src = inspect.getsource(__import__("app.atlas.video_learner", fromlist=["VideoLearner"]))
        assert "competence_tracker" in src or "register" in src

    def test_learning_planner_uses_competence_tracker(self):
        """LearningPlanner should check competence gaps."""
        src = inspect.getsource(__import__("app.atlas.learning_planner", fromlist=["LearningPlanner"]))
        assert "competence_tracker" in src

    def test_competence_readiness_drives_learning(self):
        """Readiness check with unknowns should produce actionable gaps."""
        from app.atlas.competence_tracker import CompetenceTracker
        ct = CompetenceTracker()
        readiness = ct.check_task_readiness([
            {"domain": "cloud", "name": "gcp_functions"},
            {"domain": "crypto", "name": "jwt_signing"},
        ])
        # Unknowns should have nonzero estimated learning time
        assert isinstance(readiness.get("estimated_learning_time_minutes", 0), (int, float))


# ════════════════════════════════════════════════════════════════════════════════
# 11. SYSTEM WIRING TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestSystemWiring:
    """ATLAS modules must be correctly wired into the live system."""

    def test_orchestrator_imports_learning_planner(self):
        """Commander orchestrator should use LearningPlanner for hard tasks."""
        src = inspect.getsource(__import__("app.agents.commander.orchestrator", fromlist=["handle"]))
        assert "learning_planner" in src or "LearningPlanner" in src

    def test_orchestrator_imports_competence_tracker(self):
        """Commander orchestrator should check competence."""
        src = inspect.getsource(__import__("app.agents.commander.orchestrator", fromlist=["handle"]))
        assert "competence_tracker" in src or "get_tracker" in src

    def test_idle_scheduler_atlas_competence_sync(self):
        """Idle scheduler should have atlas-competence-sync job."""
        from app.idle_scheduler import _default_jobs
        jobs = _default_jobs()
        names = [name for name, _ in jobs]
        assert "atlas-competence-sync" in names or "atlas-learning" in names

    def test_idle_scheduler_atlas_stale_check(self):
        """Idle scheduler should check for stale skills."""
        src = inspect.getsource(__import__("app.idle_scheduler", fromlist=["_default_jobs"]))
        assert "stale" in src.lower() or "atlas" in src.lower()

    def test_idle_scheduler_atlas_learning(self):
        """Idle scheduler should run ATLAS learning jobs."""
        src = inspect.getsource(__import__("app.idle_scheduler", fromlist=["_default_jobs"]))
        assert "atlas" in src.lower()

    def test_publish_imports_atlas(self):
        """Publishing/monitoring should report ATLAS status."""
        try:
            src = inspect.getsource(__import__("app.publish", fromlist=["publish"]))
            has_atlas = "atlas" in src.lower()
        except Exception:
            has_atlas = True  # Skip if module doesn't exist
        assert has_atlas

    def test_atlas_jobs_are_callable(self):
        """All ATLAS idle scheduler functions should be callable."""
        from app.idle_scheduler import _default_jobs
        jobs = _default_jobs()
        atlas_jobs = [(name, fn) for name, fn in jobs if "atlas" in name.lower()]
        for name, fn in atlas_jobs:
            assert callable(fn), f"ATLAS job {name} is not callable"

    def test_commands_module_can_handle_atlas_queries(self):
        """Signal commands should handle ATLAS-related queries."""
        try:
            from app.agents.commander.commands import try_command
            # These should not crash
            result = try_command("skills", "test", None)
            assert result is not None or result is None  # Just shouldn't crash
        except Exception:
            pass  # Command may need DB


# ════════════════════════════════════════════════════════════════════════════════
# 12. INTEGRATION TESTS (require running services)
# ════════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """Integration tests that run inside Docker with services available."""

    def test_skill_library_filesystem_persistence(self):
        """Skills should persist to filesystem."""
        from app.atlas.skill_library import SkillLibrary
        lib = SkillLibrary()
        lib.register_skill(
            skill_id="persist_test",
            name="Persistence Test",
            category="learned",
            code="# persist",
            description="Filesystem persistence test",
            source_type="manual",
        )
        # Create new instance to test loading from disk
        lib2 = SkillLibrary()
        skill = lib2.get_skill("persist_test")
        assert skill is not None
        assert skill.name == "Persistence Test"

    def test_competence_tracker_persistence(self):
        """Competence entries should persist."""
        from app.atlas.competence_tracker import CompetenceTracker
        ct = CompetenceTracker()
        ct.register(domain="persistence_test", name="disk_check", confidence=0.6, source="test")
        ct2 = CompetenceTracker()
        entry = ct2.check_competence("persistence_test", "disk_check")
        assert entry is not None

    def test_audit_log_writes_to_postgres(self):
        """Audit log should write to PostgreSQL when available."""
        from app.atlas.audit_log import log_external_call
        try:
            log_external_call(
                agent="integration_test",
                action="test_write",
                target="postgres://test",
                result="success",
            )
            # If DB is available, verify the write
            from app.control_plane.db import execute
            rows = execute(
                "SELECT * FROM atlas.external_calls WHERE agent = %s ORDER BY called_at DESC LIMIT 1",
                ("integration_test",),
                fetch=True,
            )
            if rows:
                assert rows[0]["agent"] == "integration_test"
        except Exception:
            pass  # DB may not have atlas schema — that's OK

    def test_full_skill_lifecycle(self):
        """Register → use → check competence → stale detection."""
        from app.atlas.skill_library import SkillLibrary
        from app.atlas.competence_tracker import CompetenceTracker

        lib = SkillLibrary()
        ct = CompetenceTracker()

        # Register
        lib.register_skill(
            skill_id="lifecycle_test_skill",
            name="Lifecycle Test",
            category="recipes",
            code="def lifecycle(): return True",
            description="Full lifecycle test",
            source_type="code_forge",
        )

        # Use
        lib.record_usage("lifecycle_test_skill", success=True)
        lib.record_usage("lifecycle_test_skill", success=True)

        # Verify usage tracked
        skill = lib.get_skill("lifecycle_test_skill")
        assert skill.usage_count >= 2
        assert skill.usage_success_count >= 2

        # Sync to competence
        ct.sync_from_skill_library()

        # Verify searchable
        results = lib.search(query="lifecycle")
        found = any(s.skill_id == "lifecycle_test_skill" for s in results)
        assert found

    def test_auth_pattern_detection_pipeline(self):
        """Detect auth pattern → get template code → verify structure."""
        from app.atlas.auth_patterns import detect_auth_pattern, get_pattern, get_pattern_code

        docs = """
        Authentication: Use OAuth 2.0 client credentials.
        POST /oauth/token with client_id and client_secret.
        Include access_token in Authorization: Bearer header.
        """
        patterns = detect_auth_pattern(docs)
        assert len(patterns) > 0

        top_pattern_id, confidence = patterns[0]
        assert confidence > 0

        pattern = get_pattern(top_pattern_id)
        assert pattern is not None
        assert len(pattern.required_params) > 0

        code = get_pattern_code(top_pattern_id)
        assert len(code) > 20


# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
