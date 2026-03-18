import re
from pydantic_settings import BaseSettings
from pydantic import SecretStr, field_validator
from functools import lru_cache

# Hard caps — prevent a misconfigured or tampered .env from granting runaway resources
_MAX_SANDBOX_MEMORY_MB = 2048   # 2 GB ceiling
_MIN_SANDBOX_MEMORY_MB = 32
_MAX_SANDBOX_CPU = 4.0
_MIN_SANDBOX_CPU = 0.05
_MAX_SANDBOX_TIMEOUT = 120      # seconds


class Settings(BaseSettings):
    anthropic_api_key: SecretStr
    brave_api_key: SecretStr

    signal_bot_number: str
    signal_owner_number: str
    signal_cli_path: str = "/opt/homebrew/bin/signal-cli"
    signal_socket_path: str = "/tmp/signal-cli.sock"
    # HTTP endpoint for signal-cli (preferred in Docker — Unix sockets don't work
    # across the Docker Desktop for Mac VM boundary).
    # Start signal-cli with: signal-cli daemon --http 127.0.0.1:7583 --receive-mode on-start
    # Then set SIGNAL_HTTP_URL=http://host.docker.internal:7583
    signal_http_url: str = ""
    # Path where signal-cli stores downloaded attachments.
    # On macOS: ~/.local/share/signal-cli/attachments
    # Mounted read-only into Docker at /app/attachments
    signal_attachment_path: str = ""

    gateway_secret: SecretStr
    gateway_port: int = 8765
    gateway_bind: str = "127.0.0.1"

    commander_model: str = "claude-opus-4-6"
    specialist_model: str = "claude-sonnet-4-6"

    sandbox_image: str = "crewai-sandbox:latest"
    sandbox_timeout_seconds: int = 30
    sandbox_memory_limit: str = "512m"
    sandbox_cpu_limit: float = 0.5

    self_improve_cron: str = "0 3 * * *"
    self_improve_topic_file: str = "workspace/skills/learning_queue.md"

    # Parallelism — controls how many crews/sub-agents can run concurrently.
    max_parallel_crews: int = 3   # max crews commander can dispatch at once
    max_sub_agents: int = 4       # max sub-agents a single crew can spawn
    thread_pool_size: int = 6     # shared thread pool size (caps total API calls)

    # Cloud backup — set to a GitHub/git remote URL to enable workspace sync.
    # Use an HTTPS URL with a PAT for simplicity:
    #   https://<PAT>@github.com/you/crewai-memory-backup.git
    # Leave empty to disable (the system works fine without backups).
    workspace_backup_repo: str = ""
    workspace_sync_cron: str = "0 * * * *"  # default: every hour

    # How many recent user+assistant exchanges to include in each new request
    # so the LLM understands short/contextual replies.
    conversation_history_turns: int = 10

    class Config:
        env_file = ".env"

    @field_validator("sandbox_memory_limit")
    @classmethod
    def validate_memory_limit(cls, v: str) -> str:
        """Accept Docker memory strings like 128m, 1g and clamp to safe bounds."""
        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kmgKMG]?)", v.strip())
        if not m:
            raise ValueError(f"Invalid sandbox_memory_limit format: {v!r}. Use e.g. '512m' or '1g'.")
        amount, unit = float(m.group(1)), m.group(2).lower()
        mb = amount * {"k": 1/1024, "m": 1, "g": 1024, "": 1}[unit]
        if mb < _MIN_SANDBOX_MEMORY_MB:
            raise ValueError(f"sandbox_memory_limit {v!r} is below minimum {_MIN_SANDBOX_MEMORY_MB}m.")
        if mb > _MAX_SANDBOX_MEMORY_MB:
            raise ValueError(f"sandbox_memory_limit {v!r} exceeds maximum {_MAX_SANDBOX_MEMORY_MB}m.")
        return v

    @field_validator("sandbox_cpu_limit")
    @classmethod
    def validate_cpu_limit(cls, v: float) -> float:
        if v < _MIN_SANDBOX_CPU or v > _MAX_SANDBOX_CPU:
            raise ValueError(
                f"sandbox_cpu_limit {v} out of range [{_MIN_SANDBOX_CPU}, {_MAX_SANDBOX_CPU}]."
            )
        return v

    @field_validator("sandbox_timeout_seconds")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        if v < 1 or v > _MAX_SANDBOX_TIMEOUT:
            raise ValueError(
                f"sandbox_timeout_seconds {v} out of range [1, {_MAX_SANDBOX_TIMEOUT}]."
            )
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()


# --- Scoped accessors so agents only see what they need ---

def get_anthropic_api_key() -> str:
    """Return the Anthropic API key (for LLM agents only)."""
    return get_settings().anthropic_api_key.get_secret_value()


def get_brave_api_key() -> str:
    """Return the Brave Search API key (for web search tool only)."""
    return get_settings().brave_api_key.get_secret_value()


def get_gateway_secret() -> str:
    """Return the gateway secret (for HTTP auth only)."""
    return get_settings().gateway_secret.get_secret_value()
