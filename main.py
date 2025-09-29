# main.py
"""Sparky Slack bot entry point."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

import config
from ai_parser import parse_search_query, parse_ticket_request
from branding import build_branded_blocks
from jira_client import JiraClient
from logging_utils import configure_logging
from persistence.db import Database

configure_logging(config.LOG_LEVEL, config.LOG_JSON)
logger = logging.getLogger(__name__)

app = App(token=config.SLACK_BOT_TOKEN)
jira_client = JiraClient()
database = Database()

RATE_LIMIT_SECONDS = 2
last_request_time: dict[str, int] = {}
GENERIC_ERROR_MESSAGE = (
    "Sorry, I couldn't create the Jira ticket. Please check your input or try again later."
)


def is_rate_limited(user_id):
    now = int(time.time())
    last = last_request_time.get(user_id, 0)
    if now - last < RATE_LIMIT_SECONDS:
        logger.debug("rate_limited", extra={"user_id": user_id, "last_ts": last, "now_ts": now})
        return True
    last_request_time[user_id] = now
    return False


@app.event("message")
def handle_message_events(body, say, logger):
    event = body.get("event", {})
    channel = event.get("channel")
    user = event.get("user")
    text = event.get("text", "")
    subtype = event.get("subtype")

    if _should_ignore(channel, subtype, user):
        return

    if is_rate_limited(user):
        _respond_with_logo(say, "Please wait before making another ticket request.")
        _log_interaction(
            user=user,
            channel=channel,
            message=text,
            intent="rate_limit",
            success=False,
            parser_success=True,
            parser_error=None,
            jira_error=None,
            parsed_fields=None,
            extra={"reason": "rate_limited"},
        )
        return

    if _handle_search(text, user, channel, say):
        return

    outcome = _process_ticket_request(text, user)
    if outcome.success:
        ticket_url = (
            f"https://{config.JIRA_DOMAIN}/browse/{outcome.result}" if outcome.result else None
        )
        message = (
            f"Jira ticket created: <{ticket_url}|{outcome.result}> :tada:"
            if ticket_url
            else "Jira ticket created."
        )
        _respond_with_logo(say, message)
    else:
        _respond_with_logo(say, outcome.error or GENERIC_ERROR_MESSAGE)

    _log_interaction(
        user=user,
        channel=channel,
        message=text,
        intent=outcome.intent,
        success=outcome.success,
        parser_success=outcome.parser_success,
        parser_error=outcome.parser_error,
        jira_error=outcome.jira_error,
        parsed_fields=outcome.parsed_fields,
        jira_issue_key=outcome.result,
        extra=outcome.extra,
    )


def _should_ignore(channel: Optional[str], subtype: Optional[str], user: Optional[str]) -> bool:
    if subtype == "bot_message":
        logger.debug("ignore_bot_message", extra={"user": user, "channel": channel})
        return True
    if channel != config.SLACK_CHANNEL and (not channel or not channel.startswith("D")):
        logger.debug(
            "ignore_channel",
            extra={"channel": channel, "expected_channel": config.SLACK_CHANNEL},
        )
        return True
    return False


@dataclass
class HandlerOutcome:
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None
    parser_success: bool = True
    parser_error: Optional[str] = None
    jira_error: Optional[str] = None
    parsed_fields: Optional[dict[str, Any]] = None
    intent: str = "create_issue"
    extra: dict[str, Any] = field(default_factory=dict)


def _process_ticket_request(text: str, user: Optional[str]) -> HandlerOutcome:
    try:
        parsed = parse_ticket_request(text, assignee_map=config.ASSIGNEE_MAP)
    except ValueError as err:
        logger.info("parser_rejected_message", extra={"user": user, "reason": str(err)})
        return HandlerOutcome(
            success=False,
            error=str(err),
            parser_success=False,
            parser_error=str(err),
            intent="create_issue",
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("parser_exception", extra={"user": user, "error": str(exc)})
        return HandlerOutcome(
            success=False,
            error=GENERIC_ERROR_MESSAGE,
            parser_success=False,
            parser_error=str(exc),
            intent="create_issue",
        )

    logger.info("parsed_message", extra={"user": user, "fields": sorted(parsed.keys())})
    fields = jira_client.build_fields(parsed)
    logger.debug(
        "built_fields",
        extra={
            "has_assignee": bool(fields.get("assignee")),
            "labels_count": len(fields.get("labels", [])),
        },
    )

    result = jira_client.create_issue(fields)
    if result and "key" in result:
        logger.info("ticket_created", extra={"key": result["key"], "user": user})
        return HandlerOutcome(
            success=True,
            result=result["key"],
            parsed_fields=fields,
            intent="create_issue",
            extra={"jira_response": "created"},
        )

    error_message = result.get("error") if isinstance(result, dict) else None
    if error_message:
        logger.warning("ticket_rejected", extra={"user": user, "error": error_message})
        return HandlerOutcome(
            success=False,
            error=f"Sorry, Jira rejected the request: {error_message}",
            parser_success=True,
            jira_error=error_message,
            parsed_fields=fields,
            intent="create_issue",
        )

    logger.error("ticket_creation_failed", extra={"user": user})
    return HandlerOutcome(
        success=False,
        error=GENERIC_ERROR_MESSAGE,
        parser_success=True,
        parsed_fields=fields,
        intent="create_issue",
    )


def _handle_search(text: str, user: Optional[str], channel: Optional[str], say) -> bool:
    query = parse_search_query(text)
    if not query:
        return False

    logger.info(
        "search_query",
        extra={"user": user, "jql": query.jql, "limit": query.limit, "terms": query.raw_terms},
    )

    result = jira_client.search_issues(query.jql, limit=query.limit)
    if "error" in result:
        _respond_with_logo(say, f"Search failed: {result['error']}")
        logger.warning("search_failed", extra={"user": user, "error": result["error"]})
        _log_interaction(
            user=user,
            channel=channel,
            message=text,
            intent="search_issues",
            success=False,
            parser_success=True,
            parser_error=None,
            jira_error=result["error"],
            parsed_fields={"jql": query.jql, "limit": query.limit},
            extra={"result_count": 0},
        )
        return True

    issues = result.get("issues", [])
    if not issues:
        _respond_with_logo(say, "No matching Jira issues found.")
        logger.info("search_empty", extra={"user": user})
        _log_interaction(
            user=user,
            channel=channel,
            message=text,
            intent="search_issues",
            success=True,
            parser_success=True,
            parser_error=None,
            jira_error=None,
            parsed_fields={"jql": query.jql, "limit": query.limit},
            extra={"result_count": 0},
        )
        return True

    _respond_with_logo(say, _format_search_results(issues))
    logger.info("search_results_sent", extra={"user": user, "count": len(issues)})
    _log_interaction(
        user=user,
        channel=channel,
        message=text,
        intent="search_issues",
        success=True,
        parser_success=True,
        parser_error=None,
        jira_error=None,
        parsed_fields={"jql": query.jql, "limit": query.limit},
        extra={"result_count": len(issues)},
    )
    return True


def _format_search_results(issues: list[dict[str, Optional[str]]]) -> str:
    lines = ["Here are the latest matches:"]
    for issue in issues:
        key = issue.get("key") or "Unknown"
        summary = issue.get("summary") or "No summary provided"
        status = issue.get("status") or "Unknown"
        assignee = issue.get("assignee") or "Unassigned"
        lines.append(f"- {key}: {summary} (Status: {status}, Assignee: {assignee})")
    return "\n".join(lines)


def _respond_with_logo(
    say, message: str, *, extra_blocks: Optional[list[dict[str, Any]]] = None
) -> None:
    blocks = build_branded_blocks(message, extra_blocks=extra_blocks)
    if blocks:
        say(blocks=blocks, text=message)
    else:
        say(message)


def _log_interaction(
    *,
    user: Optional[str],
    channel: Optional[str],
    message: str,
    intent: str,
    success: bool,
    parser_success: bool,
    parser_error: Optional[str],
    jira_error: Optional[str],
    parsed_fields: Optional[dict[str, Any]],
    jira_issue_key: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    try:
        database.log_interaction(
            user_id=user,
            channel=channel,
            message_text=message,
            intent=intent,
            parser_success=parser_success,
            parser_error=parser_error,
            jira_issue_key=jira_issue_key,
            jira_error=jira_error,
            parsed_fields=parsed_fields,
            extra={"success": success, **(extra or {})},
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception(
            "db_log_failed",
            extra={"intent": intent, "error": str(exc)},
        )


@app.event({"type": "message"})
def debug_all_messages(body, say, logger):
    logger.info(f"[DEBUG] Received event: {body}")
    _respond_with_logo(say, "[DEBUG] I received your message!")


if __name__ == "__main__":
    handler = SocketModeHandler(app, config.SLACK_APP_TOKEN)
    handler.start()
