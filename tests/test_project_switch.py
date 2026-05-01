"""
test_project_switch.py — Tests for project switch case-insensitivity fix.

Regression tests covering the bug where "Switch project PLG" (uppercase)
failed because:
  1. Commander command regex only matched "project switch X" word order.
  2. project_isolation.activate() did case-sensitive dict lookup.
  3. control_plane.projects.switch() passed user-cased name to activate().

Run: pytest tests/test_project_switch.py -v
"""
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ═══════════════════════════════════════════════════════════════════════════
# Command regex — accepts both word orders
# ═══════════════════════════════════════════════════════════════════════════

class TestSwitchCommandRegex:
    """The regex must match user input across:
      - Both word orders (``<noun> switch X`` AND ``switch <noun> X``)
      - Both nouns (``project`` AND ``workspace``) — the dashboard uses
        "workspace" while the DB schema uses "project"; both forms are
        valid user-facing
      - Optional ``to`` connector before OR after the noun
      - Multi-word project names (``eesti mets`` not just ``eesti``)
      - Any casing
    """

    PATTERN = re.compile(
        r"^(?:"
            r"(?:project|workspace)\s+switch"
            r"|"
            r"switch\s+(?:to\s+)?(?:project|workspace)"
        r")\s+(?:to\s+)?(.+)",
        re.IGNORECASE,
    )

    def _extract(self, text: str) -> str | None:
        m = self.PATTERN.match(text.strip())
        return m.group(1).strip().strip(".,!?") if m else None

    # ── Original "project" forms ──
    def test_legacy_word_order(self):
        assert self._extract("project switch plg") == "plg"

    def test_natural_word_order_lowercase(self):
        assert self._extract("switch project plg") == "plg"

    def test_natural_word_order_uppercase(self):
        assert self._extract("Switch project PLG") == "PLG"

    def test_with_to_connector(self):
        assert self._extract("switch to project PLG") == "PLG"

    def test_mixed_case(self):
        assert self._extract("Switch Project Archibal") == "Archibal"

    def test_strips_punctuation(self):
        assert self._extract("switch project PLG.") == "PLG"
        assert self._extract("switch project PLG!") == "PLG"

    # ── "workspace" alias (added 2026-04-30 after the agent refused
    #    "switch workspace to eesti mets" with a flustered "I can't") ──

    def test_workspace_alias_natural(self):
        assert self._extract("switch workspace plg") == "plg"

    def test_workspace_alias_with_to_after_noun(self):
        """The user's actual case: ``switch workspace TO eesti mets``."""
        assert self._extract("switch workspace to eesti mets") == "eesti mets"

    def test_workspace_alias_with_to_before_noun(self):
        assert self._extract("switch to workspace plg") == "plg"

    def test_workspace_alias_legacy_word_order(self):
        assert self._extract("workspace switch eesti mets") == "eesti mets"

    def test_workspace_alias_uppercase(self):
        assert self._extract("Switch Workspace To Eesti Mets") == "Eesti Mets"

    # ── Multi-word names — the qx \\S+ regression ──

    def test_multi_word_name_project(self):
        """Earlier regex used ``\\S+`` which truncated multi-word names."""
        assert self._extract("switch project eesti mets") == "eesti mets"

    def test_multi_word_name_three_words(self):
        assert self._extract("switch workspace to my big project") == "my big project"

    def test_multi_word_strips_trailing_punctuation(self):
        assert self._extract("switch workspace to eesti mets.") == "eesti mets"

    # ── Negative cases — must not over-match ──

    def test_rejects_unrelated_text(self):
        assert self._extract("project list") is None
        assert self._extract("workspace list") is None
        assert self._extract("what project am I on") is None
        assert self._extract("switch llm mode") is None
        assert self._extract("switch to anthropic mode") is None


# ═══════════════════════════════════════════════════════════════════════════
# Status / definitional question patterns
# ═══════════════════════════════════════════════════════════════════════════
#
# Added 2026-04-30 after the user asked "what is the current active workspace"
# and the agent routed to the research crew (no project-introspection tool)
# instead of returning the active project. Plus the agent answered "what is
# the workspace?" with hallucinated info about "UI-only switching".
#
# Both questions should match deterministic command patterns and return
# real data — no LLM round-trip, no routing.

_STATUS_PATTERN = re.compile(
    r"^(?:"
        r"what(?:'s| is)?(?:\s+the)?(?:\s+(?:current|active))*\s+(?:project|workspace)"
        r"|"
        r"which\s+(?:project|workspace)"
        r"|"
        r"(?:current|active)\s+(?:project|workspace)"
        r"|"
        r"what\s+(?:project|workspace)\s+am\s+i\s+(?:on|in|using|working)"
    r")\b",
    re.IGNORECASE,
)

_DEFINITIONAL_PATTERN = re.compile(
    r"^what(?:'s| is)?(?:\s+a)?\s+(?:project|workspace)\b",
    re.IGNORECASE,
)

_STATUS_EXACT = {
    "project status", "project",
    "workspace status", "workspace",
    "where am i",
}


def _is_status_question(text: str) -> bool:
    stripped = text.lower().rstrip("?.! ").strip()
    return stripped in _STATUS_EXACT or bool(_STATUS_PATTERN.match(stripped))


def _is_definitional_question(text: str) -> bool:
    stripped = text.lower().rstrip("?.! ").strip()
    return bool(_DEFINITIONAL_PATTERN.match(stripped))


class TestStatusQuestionPatterns:
    """Natural-language phrasings of "what is my current workspace"."""

    def test_exact_short_forms(self):
        assert _is_status_question("workspace")
        assert _is_status_question("project")
        assert _is_status_question("workspace status")
        assert _is_status_question("project status")

    def test_where_am_i(self):
        assert _is_status_question("where am I")
        assert _is_status_question("Where am I?")

    def test_what_is_current_active_workspace(self):
        """The user's actual second question."""
        assert _is_status_question("what is the current active workspace")
        assert _is_status_question("what is the current active workspace?")
        assert _is_status_question("What's the current workspace?")

    def test_which_workspace(self):
        assert _is_status_question("which workspace")
        assert _is_status_question("which project am I on")
        assert _is_status_question("Which workspace are we using?")

    def test_what_project_am_i_on(self):
        assert _is_status_question("what project am I on")
        assert _is_status_question("what workspace am I working on")
        assert _is_status_question("what workspace am I in")

    def test_current_workspace_short(self):
        assert _is_status_question("current workspace")
        assert _is_status_question("active project")

    def test_strips_trailing_punctuation(self):
        assert _is_status_question("workspace?")
        assert _is_status_question("workspace.")
        assert _is_status_question("workspace!")

    def test_does_not_match_definitional_question(self):
        """Definitional ('what is a workspace') should fall to its own
        handler, not the status one — the answer shape is different."""
        # A bare "what is a workspace" doesn't talk about THE/current/active
        # workspace, so the status pattern shouldn't fire.
        assert not _is_status_question("what is a workspace")

    def test_does_not_match_unrelated(self):
        assert not _is_status_question("switch workspace to plg")
        assert not _is_status_question("workspaces")  # plural is the LIST command
        assert not _is_status_question("hello")


class TestDefinitionalQuestionPatterns:
    """Natural-language phrasings of "what is a workspace".

    Note: "what is THE workspace" intentionally falls to the STATUS
    pattern (returns current active) — the definite article reads as
    "the [current] workspace". Only "what is A workspace" or "what is
    workspace" routes to definitional. The status response already
    includes a "Switch with..." hint so the user gets the same
    actionable answer either way.
    """

    def test_what_is_a_workspace(self):
        assert _is_definitional_question("what is a workspace")
        assert _is_definitional_question("What's a project?")

    def test_what_is_workspace_no_article(self):
        """Bare "what is workspace" → definitional (no article = generic)."""
        assert _is_definitional_question("what is workspace")
        assert _is_definitional_question("what is project")

    def test_what_is_THE_workspace_is_NOT_definitional(self):
        """Definite article reads as "the current" — should fall to
        status pattern, not definitional. Status pattern test covers
        the positive case in TestStatusQuestionPatterns."""
        assert not _is_definitional_question("what is the workspace")
        assert _is_status_question("what is the workspace")

    def test_does_not_match_unrelated(self):
        assert not _is_definitional_question("switch workspace to plg")
        assert not _is_definitional_question("hello")
        assert not _is_definitional_question("which workspace")


# ═══════════════════════════════════════════════════════════════════════════
# project_isolation.ProjectManager — case-insensitive activate()
# ═══════════════════════════════════════════════════════════════════════════

class TestProjectIsolationCaseInsensitivity:

    def _make_manager(self, tmp_path):
        from app.project_isolation import ProjectManager, ProjectConfig
        pm = ProjectManager(projects_dir=tmp_path)
        pm._projects = {
            "plg": ProjectConfig(name="plg", display_name="Protect Group"),
            "archibal": ProjectConfig(name="archibal", display_name="Archibal"),
            "kaicart": ProjectConfig(name="kaicart", display_name="KaiCart"),
        }
        # Create empty project dirs so activate() doesn't fail on missing paths
        for name in pm._projects:
            (tmp_path / name / "instructions").mkdir(parents=True, exist_ok=True)
            (tmp_path / name / "variables.env").write_text("")
        return pm

    def test_resolve_name_exact_match(self, tmp_path):
        pm = self._make_manager(tmp_path)
        assert pm._resolve_name("plg") == "plg"

    def test_resolve_name_uppercase(self, tmp_path):
        pm = self._make_manager(tmp_path)
        assert pm._resolve_name("PLG") == "plg"

    def test_resolve_name_mixed_case(self, tmp_path):
        pm = self._make_manager(tmp_path)
        assert pm._resolve_name("Plg") == "plg"
        assert pm._resolve_name("ArChIbAl") == "archibal"

    def test_resolve_name_unknown(self, tmp_path):
        pm = self._make_manager(tmp_path)
        assert pm._resolve_name("nonexistent") is None

    def test_activate_uppercase_succeeds(self, tmp_path):
        pm = self._make_manager(tmp_path)
        ctx = pm.activate("PLG")
        assert ctx.name == "plg"
        assert pm.active.name == "plg"

    def test_activate_mixed_case_succeeds(self, tmp_path):
        pm = self._make_manager(tmp_path)
        ctx = pm.activate("ArChIbAl")
        assert ctx.name == "archibal"

    def test_activate_unknown_raises(self, tmp_path):
        pm = self._make_manager(tmp_path)
        with pytest.raises(ValueError, match="Unknown project"):
            pm.activate("DOES_NOT_EXIST")


# ═══════════════════════════════════════════════════════════════════════════
# control_plane.projects.switch — canonical name propagation
# ═══════════════════════════════════════════════════════════════════════════

class TestControlPlaneSwitchCanonical:

    def test_switch_passes_canonical_name_to_isolation(self):
        """switch('PLG') should call project_isolation.activate('plg')."""
        from app.control_plane.projects import ProjectManager

        manager = ProjectManager()
        # Mock the database lookup to return a canonical lowercase row
        canonical_row = {
            "id": "abc-123",
            "name": "plg",
            "mission": "Live entertainment ticketing",
        }

        with patch.object(manager, "get_by_name", return_value=canonical_row), \
             patch("app.project_isolation.get_manager") as mock_get_pm:
            mock_pm = MagicMock()
            mock_get_pm.return_value = mock_pm

            result = manager.switch("PLG")  # user-supplied uppercase

            assert result == canonical_row
            # Verify activate was called with the CANONICAL lowercase name
            mock_pm.activate.assert_called_once_with("plg")

    def test_switch_returns_none_for_unknown(self):
        from app.control_plane.projects import ProjectManager
        manager = ProjectManager()
        with patch.object(manager, "get_by_name", return_value=None):
            result = manager.switch("GHOST")
        assert result is None

    def test_switch_tolerates_activate_failure(self):
        """If project_isolation fails, control_plane switch still succeeds."""
        from app.control_plane.projects import ProjectManager
        manager = ProjectManager()
        canonical_row = {"id": "x", "name": "plg", "mission": "m"}

        with patch.object(manager, "get_by_name", return_value=canonical_row), \
             patch("app.project_isolation.get_manager", side_effect=Exception("boom")):
            result = manager.switch("PLG")

        # Control-plane switch still returns the project row
        assert result == canonical_row
        assert manager._active_project_id == "x"
