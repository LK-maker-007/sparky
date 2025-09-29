"""Compatibility layer exposing Pydantic-based settings."""

from functools import lru_cache
from typing import Dict, Optional

from settings import SparkySettings, get_settings


@lru_cache(maxsize=1)
def _cached_settings() -> SparkySettings:
    return get_settings()


def get_slack_bot_token() -> str:
    return _cached_settings().slack.bot_token


def get_slack_app_token() -> str:
    return _cached_settings().slack.app_token


def get_slack_channel() -> str:
    return _cached_settings().slack.default_channel


def get_jira_domain() -> str:
    return _cached_settings().jira.domain


def get_jira_email() -> str:
    return _cached_settings().jira.email


def get_jira_api_token() -> str:
    return _cached_settings().jira.api_token


def get_jira_project_key() -> str:
    return _cached_settings().jira.project_key


def get_jira_default_issue_type() -> str:
    return _cached_settings().jira.default_issue_type


def get_assignee_map() -> Dict[str, str]:
    return _cached_settings().assignee_map


def get_default_assignee_account_id():
    return _cached_settings().jira.default_account_id


def is_text_rewrite_enabled() -> bool:
    return _cached_settings().ollama.enable_text_rewrite


def get_ollama_host() -> str:
    return _cached_settings().ollama.host


def get_ollama_model() -> str:
    return _cached_settings().ollama.model


def get_ollama_timeout() -> int:
    return _cached_settings().ollama.timeout_seconds


def is_intent_detection_enabled() -> bool:
    return _cached_settings().ollama.enable_intent_detection


def get_logo_path() -> Optional[str]:
    return _cached_settings().branding.logo_path


def get_logo_url() -> Optional[str]:
    return _cached_settings().branding.logo_url


def get_log_level() -> str:
    return _cached_settings().logging.level


def is_log_json_enabled() -> bool:
    return _cached_settings().logging.json_enabled


# Backwards-compatible module-level constants
SLACK_BOT_TOKEN = get_slack_bot_token()
SLACK_APP_TOKEN = get_slack_app_token()
SLACK_CHANNEL = get_slack_channel()

JIRA_DOMAIN = get_jira_domain()
JIRA_EMAIL = get_jira_email()
JIRA_API_TOKEN = get_jira_api_token()
JIRA_PROJECT_KEY = get_jira_project_key()
JIRA_DEFAULT_ISSUE_TYPE = get_jira_default_issue_type()

ASSIGNEE_MAP = get_assignee_map()
DEFAULT_ASSIGNEE_ACCOUNT_ID = get_default_assignee_account_id()

ENABLE_TEXT_REWRITE = is_text_rewrite_enabled()
OLLAMA_HOST = get_ollama_host()
OLLAMA_MODEL = get_ollama_model()
OLLAMA_TIMEOUT = get_ollama_timeout()
ENABLE_INTENT_DETECTION = is_intent_detection_enabled()
LOG_LEVEL = get_log_level()
LOG_JSON = is_log_json_enabled()
LOGO_PATH = get_logo_path()
LOGO_URL = get_logo_url()

__all__ = [
    "SLACK_BOT_TOKEN",
    "SLACK_APP_TOKEN",
    "SLACK_CHANNEL",
    "JIRA_DOMAIN",
    "JIRA_EMAIL",
    "JIRA_API_TOKEN",
    "JIRA_PROJECT_KEY",
    "JIRA_DEFAULT_ISSUE_TYPE",
    "ASSIGNEE_MAP",
    "DEFAULT_ASSIGNEE_ACCOUNT_ID",
    "ENABLE_TEXT_REWRITE",
    "OLLAMA_HOST",
    "OLLAMA_MODEL",
    "OLLAMA_TIMEOUT",
    "ENABLE_INTENT_DETECTION",
    "LOGO_PATH",
    "LOGO_URL",
    "LOG_LEVEL",
    "LOG_JSON",
    "get_log_level",
    "is_log_json_enabled",
    "get_settings",
]
