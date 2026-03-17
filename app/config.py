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
