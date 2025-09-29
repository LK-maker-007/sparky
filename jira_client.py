# jira_client.py
"""Jira API client helpers for Sparky."""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional, cast

import requests  # type: ignore[import-untyped]

import config

JIRA_API_URL = f"https://{config.JIRA_DOMAIN}/rest/api/3/issue"
JIRA_SEARCH_URL = f"https://{config.JIRA_DOMAIN}/rest/api/3/search"

logger = logging.getLogger(__name__)


class JiraClient:
    def __init__(self) -> None:
        self.auth = (config.JIRA_EMAIL, config.JIRA_API_TOKEN)
        self.headers: dict[str, str] = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def create_issue(
        self, fields: dict[str, Any], max_retries: int = 3, backoff: float = 2.0
    ) -> Optional[dict[str, Any]]:
        payload = {"fields": _sanitize_fields(fields)}
        response = self._post_with_retries(payload, max_retries=max_retries, backoff=backoff)
        if response is None:
            return {"error": "Failed to create Jira issue after retries."}

        if response.status_code == 201:
            logger.info("jira_issue_created", extra={"status": response.status_code})
            return cast(dict[str, Any], response.json())

        return self._handle_error(response)

    def build_fields(self, parsed: dict[str, Any]) -> dict[str, Any]:
        summary = parsed.get("summary", "[No summary provided]") or "[No summary provided]"
        summary = re.sub(r"\s+", " ", summary).strip()

        issue_type_name = parsed.get("issue_type") or config.JIRA_DEFAULT_ISSUE_TYPE or "Task"

        fields: dict[str, Any] = {
            "project": {"key": config.JIRA_PROJECT_KEY},
            "summary": summary,
            "description": _format_description(parsed.get("description")),
            "issuetype": {"name": issue_type_name},
        }
        if parsed.get("assignee"):
            fields["assignee"] = {"accountId": parsed["assignee"]}
        if "labels" in parsed:
            fields["labels"] = cast(list[str], parsed["labels"])
        return fields

    def _post_with_retries(
        self, payload: dict[str, Any], *, max_retries: int, backoff: float
    ) -> Optional[requests.Response]:
        delay = backoff
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    JIRA_API_URL,
                    json=payload,
                    headers=self.headers,
                    auth=self.auth,
                    timeout=10,
                )
                if response.status_code != 429:
                    return response

                logger.warning(
                    "jira_rate_limited",
                    extra={"attempt": attempt, "retry_in_seconds": delay},
                )
                time.sleep(delay)
                delay *= 2
            except requests.RequestException as exc:
                logger.error("jira_request_failed", extra={"attempt": attempt, "error": str(exc)})
                time.sleep(delay)
                delay *= 2
        return None

    def _handle_error(self, response: requests.Response) -> dict[str, Any]:
        try:
            payload_obj = response.json()
        except ValueError:
            message = f"Jira error {response.status_code}: {response.text}"
            logger.error(
                "jira_error_response",
                extra={"status": response.status_code, "body": response.text},
            )
            return {"error": message}

        if not isinstance(payload_obj, dict):
            logger.error(
                "jira_error_non_dict_response",
                extra={"status": response.status_code, "body_type": type(payload_obj).__name__},
            )
            return {
                "error": f"Unexpected response format from Jira ({type(payload_obj).__name__})."
            }

        messages: list[str] = []
        error_messages_val = payload_obj.get("errorMessages", [])
        if isinstance(error_messages_val, list):
            messages.extend(str(msg) for msg in error_messages_val)

        field_errors_val = payload_obj.get("errors", {})
        if isinstance(field_errors_val, dict):
            for field, msg in field_errors_val.items():
                messages.append(f"{field}: {msg}")

        message = (
            f"Jira rejected the request: {'; '.join(messages)}"
            if messages
            else f"Jira error {response.status_code}: {response.text}"
        )
        logger.error(
            "jira_validation_failed",
            extra={"status": response.status_code, "errors": messages},
        )
        return {"error": message}

    def search_issues(self, jql: str, limit: int = 5) -> dict[str, Any]:
        params = {
            "jql": jql,
            "maxResults": limit,
            "fields": "summary,status,assignee",
        }
        try:
            response = requests.get(
                JIRA_SEARCH_URL,
                params=params,
                headers=self.headers,
                auth=self.auth,
                timeout=10,
            )
        except requests.RequestException as exc:
            logger.error("jira_search_failed", extra={"error": str(exc), "jql": jql})
            return {"error": str(exc)}

        if response.status_code != 200:
            logger.error(
                "jira_search_error",
                extra={"status": response.status_code, "body": response.text, "jql": jql},
            )
            return {"error": f"Search failed with status {response.status_code}"}

        try:
            payload_obj = response.json()
        except ValueError as exc:
            logger.error("jira_search_invalid_json", extra={"error": str(exc)})
            return {"error": "Invalid response from Jira search."}

        if not isinstance(payload_obj, dict):
            logger.error(
                "jira_search_unexpected_format",
                extra={"body_type": type(payload_obj).__name__},
            )
            return {"error": "Unexpected response format from Jira search."}

        issues_raw = payload_obj.get("issues", [])
        if not isinstance(issues_raw, list):
            issues_raw = []

        issues = [_map_issue_summary(issue) for issue in issues_raw if isinstance(issue, dict)]
        return {"issues": issues}


def _sanitize_fields(fields: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in fields.items() if value not in (None, [], "")}


def _format_description(text: Optional[str]) -> dict[str, Any]:
    safe_text = text or "[No description provided]"
    return {
        "type": "doc",
        "version": 1,
        "content": [{"type": "paragraph", "content": [{"type": "text", "text": safe_text}]}],
    }


def _map_issue_summary(issue: dict[str, Any]) -> dict[str, Optional[str]]:
    fields = cast(dict[str, Any], issue.get("fields", {}))
    status = cast(dict[str, Any], fields.get("status", {}))
    assignee = cast(dict[str, Any], fields.get("assignee", {}))
    return {
        "key": cast(Optional[str], issue.get("key")),
        "summary": cast(Optional[str], fields.get("summary")),
        "status": cast(Optional[str], status.get("name")),
        "assignee": cast(Optional[str], assignee.get("displayName")),
    }
