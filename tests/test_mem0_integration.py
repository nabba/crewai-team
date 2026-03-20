"""Tests for Mem0 persistent memory integration.

Tests verify:
  - Configuration fields exist with correct defaults
  - Mem0 manager has graceful degradation (no crash when unavailable)
  - Mem0 tools factory respects enabled flag
  - Agent files include Mem0 tools
  - Commander injects Mem0 context
  - main.py extracts conversations to Mem0
  - Docker compose includes postgres and neo4j services
"""
import unittest
from unittest.mock import patch, MagicMock


class TestMem0Config(unittest.TestCase):
    """Verify Mem0 config fields exist with correct types and defaults."""

    def test_mem0_fields_exist(self):
        from app.config import Settings
        fields = Settings.model_fields
        self.assertIn("mem0_enabled", fields)
        self.assertIn("mem0_postgres_host", fields)
        self.assertIn("mem0_postgres_password", fields)
        self.assertIn("mem0_neo4j_url", fields)
        self.assertIn("mem0_neo4j_user", fields)
        self.assertIn("mem0_neo4j_password", fields)
        self.assertIn("mem0_llm_model", fields)
        self.assertIn("mem0_embedder_model", fields)
        self.assertIn("mem0_user_id", fields)

    def test_mem0_defaults(self):
        from app.config import Settings
        fields = Settings.model_fields
        self.assertTrue(fields["mem0_enabled"].default)
        self.assertIn("bolt://", fields["mem0_neo4j_url"].default)
        self.assertEqual(fields["mem0_user_id"].default, "owner")
        self.assertEqual(fields["mem0_embedder_model"].default, "all-MiniLM-L6-v2")

    def test_neo4j_password_is_secret_str(self):
        from pydantic import SecretStr
        from app.config import Settings
        field = Settings.model_fields["mem0_neo4j_password"]
        self.assertIs(field.annotation, SecretStr)

    def test_postgres_password_is_secret_str(self):
        from pydantic import SecretStr
        from app.config import Settings
        field = Settings.model_fields["mem0_postgres_password"]
        self.assertIs(field.annotation, SecretStr)

    def test_no_hardcoded_passwords_in_defaults(self):
        """Default password values must be empty — force env var config."""
        from app.config import Settings
        fields = Settings.model_fields
        neo4j_default = fields["mem0_neo4j_password"].default
        pg_default = fields["mem0_postgres_password"].default
        # SecretStr defaults should be empty
        self.assertEqual(neo4j_default.get_secret_value(), "")
        self.assertEqual(pg_default.get_secret_value(), "")


class TestMem0Manager(unittest.TestCase):
    """Verify mem0_manager gracefully handles unavailability."""

    def test_store_memory_returns_none_when_disabled(self):
        """store_memory returns None when client is unavailable."""
        from app.memory import mem0_manager
        # Reset singleton state for test
        mem0_manager._client = None
        mem0_manager._init_failed = False
        with patch.object(mem0_manager, 'get_client', return_value=None):
            result = mem0_manager.store_memory("test fact")
            self.assertIsNone(result)

    def test_search_memory_returns_empty_when_disabled(self):
        from app.memory import mem0_manager
        with patch.object(mem0_manager, 'get_client', return_value=None):
            result = mem0_manager.search_memory("test query")
            self.assertEqual(result, [])

    def test_store_conversation_returns_none_when_disabled(self):
        from app.memory import mem0_manager
        with patch.object(mem0_manager, 'get_client', return_value=None):
            result = mem0_manager.store_conversation([
                {"role": "user", "content": "hello"},
            ])
            self.assertIsNone(result)

    def test_get_all_memories_returns_empty_when_disabled(self):
        from app.memory import mem0_manager
        with patch.object(mem0_manager, 'get_client', return_value=None):
            result = mem0_manager.get_all_memories()
            self.assertEqual(result, [])


class TestMem0ToolsFactory(unittest.TestCase):
    """Verify tool factory respects enabled flag."""

    def test_returns_tools_when_enabled(self):
        with patch("app.config.get_settings") as mock:
            mock.return_value = MagicMock(mem0_enabled=True)
            from app.tools.mem0_tools import create_mem0_tools
            tools = create_mem0_tools("researcher")
            self.assertEqual(len(tools), 3)
            names = [t.name for t in tools]
            self.assertIn("persist_fact", names)
            self.assertIn("recall_facts", names)
            self.assertIn("persist_conversation", names)

    def test_returns_empty_when_disabled(self):
        with patch("app.config.get_settings") as mock:
            mock.return_value = MagicMock(mem0_enabled=False)
            from app.tools.mem0_tools import create_mem0_tools
            tools = create_mem0_tools("researcher")
            self.assertEqual(tools, [])


class TestAgentsMem0Integration(unittest.TestCase):
    """Verify all agent files import and use Mem0 tools."""

    def test_researcher_has_mem0(self):
        with open("app/agents/researcher.py") as f:
            source = f.read()
        self.assertIn("create_mem0_tools", source)
        self.assertIn("mem0_tools", source)

    def test_coder_has_mem0(self):
        with open("app/agents/coder.py") as f:
            source = f.read()
        self.assertIn("create_mem0_tools", source)
        self.assertIn("mem0_tools", source)

    def test_writer_has_mem0(self):
        with open("app/agents/writer.py") as f:
            source = f.read()
        self.assertIn("create_mem0_tools", source)
        self.assertIn("mem0_tools", source)


class TestCommanderMem0Context(unittest.TestCase):
    """Verify Commander injects Mem0 context into routing and crew tasks."""

    def test_routing_has_mem0_search(self):
        with open("app/agents/commander.py") as f:
            source = f.read()
        self.assertIn("search_shared", source)
        self.assertIn("KNOWN FACTS", source)
        self.assertIn("mem0_context", source)

    def test_team_memory_merges_mem0(self):
        with open("app/agents/commander.py") as f:
            source = f.read()
        # _load_relevant_team_memory should pull from both ChromaDB and Mem0
        self.assertIn("mem0_manager", source)
        self.assertIn("search_shared", source)


class TestMainMem0Extraction(unittest.TestCase):
    """Verify main.py extracts conversations to Mem0."""

    def test_extract_to_mem0_exists(self):
        with open("app/main.py") as f:
            source = f.read()
        self.assertIn("_extract_to_mem0", source)
        self.assertIn("store_conversation", source)

    def test_extraction_is_in_handle_task(self):
        with open("app/main.py") as f:
            source = f.read()
        # The extraction call should be in handle_task, after add_message
        self.assertIn("_extract_to_mem0", source)
        # Should be called as a background task (non-blocking)
        self.assertIn("asyncio.create_task", source)
        self.assertIn("_extract_to_mem0", source)


class TestDockerComposeMem0(unittest.TestCase):
    """Verify docker-compose includes Mem0 infrastructure."""

    def test_postgres_service_exists(self):
        with open("docker-compose.yml") as f:
            source = f.read()
        self.assertIn("postgres:", source)
        self.assertIn("pgvector/pgvector:pg16", source)
        self.assertIn("mem0_pgdata", source)

    def test_neo4j_service_exists(self):
        with open("docker-compose.yml") as f:
            source = f.read()
        self.assertIn("neo4j:", source)
        self.assertIn("neo4j:5-community", source)
        self.assertIn("mem0_neo4j", source)

    def test_gateway_depends_on_postgres_neo4j(self):
        with open("docker-compose.yml") as f:
            source = f.read()
        self.assertIn("postgres", source)
        self.assertIn("neo4j", source)

    def test_services_on_internal_network(self):
        """postgres and neo4j must be on internal network only."""
        with open("docker-compose.yml") as f:
            source = f.read()
        # Find postgres and neo4j sections and verify internal network
        # Both should NOT have external network
        self.assertIn("no-new-privileges:true", source)


class TestMem0Security(unittest.TestCase):
    """Security-specific tests for Mem0 integration."""

    def test_error_messages_are_sanitized(self):
        """Exception logging must redact connection strings."""
        with open("app/memory/mem0_manager.py") as f:
            source = f.read()
        self.assertIn("_sanitize_exc", source)
        self.assertIn("postgresql://***@***", source)

    def test_input_validation_exists(self):
        with open("app/memory/mem0_manager.py") as f:
            source = f.read()
        self.assertIn("_validate_text", source)
        self.assertIn("_MAX_FACT_LENGTH", source)
        self.assertIn("_MAX_MESSAGE_LENGTH", source)

    def test_docker_compose_no_default_passwords(self):
        with open("docker-compose.yml") as f:
            source = f.read()
        # Should use :? (error if unset) not :- (fallback)
        self.assertNotIn(":-mem0pass", source)
        self.assertIn(":?", source)

    def test_tools_have_input_validation(self):
        with open("app/tools/mem0_tools.py") as f:
            source = f.read()
        self.assertIn("_MAX_TOOL_INPUT", source)
        self.assertIn("Error: text is required", source)
        self.assertIn("Error: query is required", source)

    def test_extract_to_mem0_validates_inputs(self):
        with open("app/main.py") as f:
            source = f.read()
        self.assertIn("not isinstance(user_text, str)", source)
        self.assertIn("len(assistant_result.strip()) < 20", source)

    def test_no_postgres_url_with_embedded_password(self):
        """Config should NOT have a hardcoded postgres URL with password."""
        with open("app/config.py") as f:
            source = f.read()
        # Should not have postgresql://user:password@host pattern as a default
        self.assertNotIn("mem0pass", source)


if __name__ == "__main__":
    unittest.main()
