# settings.py
"""Centralized configuration using Pydantic settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings

load_dotenv()


class SlackSettings(BaseSettings):
    bot_token: str = Field(alias="SLACK_BOT_TOKEN")
    app_token: str = Field(alias="SLACK_APP_TOKEN")
    default_channel: str = Field(alias="SLACK_CHANNEL")


class JiraSettings(BaseSettings):
    domain: str = Field(alias="JIRA_DOMAIN")
    email: str = Field(alias="JIRA_EMAIL")
    api_token: str = Field(alias="JIRA_API_TOKEN")
    project_key: str = Field(alias="JIRA_PROJECT_KEY", default="S1")
    default_issue_type: str = Field(alias="JIRA_DEFAULT_ISSUE_TYPE", default="Task")
    default_account_id: Optional[str] = Field(alias="JIRA_DEFAULT_ACCOUNT_ID", default=None)


class OllamaSettings(BaseSettings):
    enable_text_rewrite: bool = Field(alias="ENABLE_TEXT_REWRITE", default=False)
    enable_intent_detection: bool = Field(alias="ENABLE_INTENT_DETECTION", default=False)
    host: str = Field(alias="OLLAMA_HOST", default="http://localhost:11434")
    model: str = Field(alias="OLLAMA_MODEL", default="llama3")
    timeout_seconds: int = Field(alias="OLLAMA_TIMEOUT", default=30)


class LoggingSettings(BaseSettings):
    level: str = Field(alias="SPARKY_LOG_LEVEL", default="INFO")
    json_enabled: bool = Field(alias="SPARKY_LOG_JSON", default=False)


class BrandingSettings(BaseSettings):
    logo_path: Optional[str] = Field(alias="SPARKY_LOGO_PATH", default="assets/sparky_logo.png")
    logo_url: Optional[str] = Field(alias="SPARKY_LOGO_URL", default=None)


class SparkySettings(BaseSettings):
    slack: SlackSettings
    jira: JiraSettings
    ollama: OllamaSettings
    logging: LoggingSettings
    branding: BrandingSettings
    assignee_map: dict[str, str] = Field(default_factory=dict)

    @staticmethod
    def _build_assignee_map() -> dict[str, str]:
        from os import environ

        mapping: dict[str, str] = {}
        prefix = "JIRA_ACCOUNT_ID_"
        for env_key, value in environ.items():
            if env_key.startswith(prefix) and value:
                alias = env_key[len(prefix) :].lower()
                alias_spaces = alias.replace("_", " ")
                mapping[alias] = value
                mapping[alias_spaces] = value
        return mapping

    @classmethod
    def load(cls) -> SparkySettings:
        try:
            settings = cls(
                slack=SlackSettings(),  # type: ignore[call-arg]
                jira=JiraSettings(),  # type: ignore[call-arg]
                ollama=OllamaSettings(),  # type: ignore[call-arg]
                logging=LoggingSettings(),  # type: ignore[call-arg]
                branding=BrandingSettings(),  # type: ignore[call-arg]
                assignee_map=cls._build_assignee_map(),
            )
        except ValidationError as exc:  # pragma: no cover - surfaced on startup
            # Re-raise with a friendlier message
            missing = [error["loc"][0] for error in exc.errors() if error.get("type") == "missing"]
            msg = "Missing required configuration values: " + ", ".join(
                sorted({str(loc) for loc in missing})
            )
            raise RuntimeError(msg) from exc

        default_id = settings.jira.default_account_id
        if default_id:
            for alias in ("me", "myself", "self"):
                settings.assignee_map.setdefault(alias, default_id)

        if not default_id and settings.assignee_map:
            settings.jira.default_account_id = next(iter(settings.assignee_map.values()))
            for alias in ("me", "myself", "self"):
                settings.assignee_map.setdefault(alias, settings.jira.default_account_id)

        return settings


@lru_cache(maxsize=1)
def get_settings() -> SparkySettings:
    return SparkySettings.load()
