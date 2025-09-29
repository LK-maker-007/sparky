# ai_parser.py
"""
AI parser for mapping natural language requests to Jira fields via local LLM helpers.
"""

import difflib
import logging
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set

import config
from llm_client import OllamaClient

logger = logging.getLogger(__name__)

FILLER_TOKENS: Set[str] = {
    "hey",
    "hi",
    "hello",
    "pls",
    "please",
    "kindly",
    "thanks",
    "make",
    "create",
    "open",
    "file",
    "raise",
    "log",
    "need",
    "want",
    "do",
    "build",
    "setup",
    "set",
    "up",
    "with",
    "fix",
    "bug",
    "ticket",
    "issue",
    "story",
    "task",
    "item",
    "this",
    "an",
    "a",
    "the",
    "asap",
}

STOP_TOKENS: List[str] = [
    "priority",
    "assign",
    "tag",
    "tags",
    "label",
    "labels",
    "due",
    "deadline",
    "title",
    "description",
]

GREETING_PATTERNS: List[str] = [
    "hi",
    "hello",
    "hey",
    "how are you",
    "how are u",
    "good morning",
    "good afternoon",
    "good evening",
]

CASUAL_TOKENS: Set[str] = {"how", "are", "you", "u", "bro", "man", "there"}

SEARCH_KEYWORDS: Set[str] = {"search", "find", "show", "list", "lookup"}
MY_KEYWORDS: Set[str] = {"my", "me", "assigned", "mine"}
OPEN_KEYWORDS: Set[str] = {"open", "pending", "active", "todo", "unresolved", "backlog"}
SEARCH_STOPWORDS: Set[str] = {
    "issues",
    "issue",
    "tickets",
    "ticket",
    "for",
    "please",
    "jira",
    "with",
    "the",
    "latest",
}
ISSUE_KEY_PATTERN = re.compile(r"\b([A-Z][A-Z0-9]+-\d+)\b", re.IGNORECASE)

FORM_SECTION_HEADERS: Set[str] = {
    "description",
    "subtasks",
    "linked work items",
    "confluence content",
    "known errors",
    "pinned fields",
    "details",
    "assignee",
    "labels",
    "parent",
    "due date",
    "team",
}
FORM_SECTION_SKIP: Set[str] = {
    "add epic",
    "add subtask",
    "add linked work item",
    "add work item",
    "improve work item",
    "try template",
    "pinned fields",
}
STATUS_KEYWORDS: Set[str] = {"to do", "in progress", "done", "ready", "blocked"}

_ollama_client: Optional[OllamaClient] = None


@dataclass
class ParsedTicket:
    issue_type: str
    summary: str
    description: str
    assignee: Optional[str]
    labels: List[str]
    priority: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if data.get("priority") is None:
            data.pop("priority", None)
        return data


@dataclass
class SearchQuery:
    jql: str
    limit: int = 5
    raw_terms: Optional[str] = None


class TicketParser:
    """Encapsulates parsing logic for a single message."""

    def __init__(self, message: str, assignee_map: Optional[Dict[str, str]] = None) -> None:
        self.original_message = message
        self.message = _normalize_common_typos(message)
        self.assignee_map = assignee_map or {}
        self.assignee_name: Optional[str] = None
        self.assignee_account: Optional[str] = None
        self.clean_message = self.message
        self.structured_summary_hint: Optional[str] = None
        self.structured_description_hint: Optional[str] = None

    def parse(self) -> ParsedTicket:
        logger.info("parsing_message", extra={"raw_message": self.original_message})

        (
            self.structured_summary_hint,
            self.structured_description_hint,
        ) = _extract_structured_form_sections(self.message)

        self.assignee_name = _extract_assignee(self.message)
        self.assignee_account = self._resolve_assignee(self.assignee_name)
        self.clean_message = _remove_assignee_phrases(self.message, self.assignee_name)

        intent = self._detect_intent()
        if intent == "greeting":
            logger.info("message_ignored", extra={"reason": "intent_greeting"})
            raise ValueError("Say a bit more about the issue so I can help create a ticket.")

        if intent is None and self._is_casual_greeting():
            logger.info("message_ignored", extra={"reason": "casual_greeting"})
            raise ValueError("Say a bit more about the issue so I can help create a ticket.")

        if not self._has_meaningful_content():
            logger.info("message_ignored", extra={"reason": "no_meaningful_tokens"})
            raise ValueError("Please provide some details about the ticket you want to create.")

        title = _extract_title(self.clean_message)
        summary = _extract_summary(
            self.clean_message,
            explicit_title=title,
            structured_hint=self.structured_summary_hint,
        )
        issue_type = (
            _extract_issue_type(self.clean_message) or config.JIRA_DEFAULT_ISSUE_TYPE or "Task"
        )
        priority = _extract_priority(self.clean_message)
        description = self._build_description()
        labels = _extract_labels(self.clean_message)

        ticket = ParsedTicket(
            issue_type=issue_type,
            summary=summary,
            description=description,
            assignee=self.assignee_account,
            labels=labels,
            priority=priority,
        )
        logger.debug(
            "parsed_ticket_summary",
            extra={"issue_type": issue_type, "has_priority": bool(priority), "labels": len(labels)},
        )
        return ticket

    def _resolve_assignee(self, assignee_name: Optional[str]) -> Optional[str]:
        if not assignee_name:
            return None
        key = assignee_name.lower().strip()
        candidates = [key]
        if " " in key:
            parts = key.split()
            candidates.extend(parts)
            candidates.append("".join(parts))
            candidates.append("_".join(parts))

        seen: Set[str] = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            account = self.assignee_map.get(candidate)
            if account:
                logger.debug(
                    "assignee_resolved",
                    extra={"requested": assignee_name, "matched_alias": candidate},
                )
                return account

        logger.warning(
            "assignee_not_found",
            extra={"assignee": assignee_name, "available": bool(self.assignee_map)},
        )
        return None

    def _has_meaningful_content(self) -> bool:
        tokens = _normalize_tokens(self.clean_message)
        return any(token not in FILLER_TOKENS for token in tokens)

    def _detect_intent(self) -> Optional[str]:
        if not config.ENABLE_INTENT_DETECTION:
            return None
        client = _ensure_ollama_client()
        if not client:
            return None
        label = client.detect_intent(self.clean_message)
        if label:
            logger.debug("intent_detected", extra={"intent": label})
        return label

    def _is_casual_greeting(self) -> bool:
        lowered = self.clean_message.strip().lower()
        for pattern in GREETING_PATTERNS:
            if lowered == pattern or lowered.startswith(pattern + " "):
                return True
        tokens = _normalize_tokens(lowered)
        return len(tokens) <= 4 and all(token in CASUAL_TOKENS for token in tokens)

    def _build_description(self) -> str:
        explicit_description = _extract_description(
            self.clean_message, structured_hint=self.structured_description_hint
        )
        if explicit_description:
            base_text = explicit_description
        else:
            ticket_phrase = _extract_ticket_phrase(self.clean_message)
            remainder = self.clean_message
            if ticket_phrase:
                remainder = remainder.replace(ticket_phrase, "", 1)
            remainder = _strip_at_stop(remainder)
            base_text = remainder or ticket_phrase or self.clean_message
            if _is_mostly_filler(base_text) and ticket_phrase:
                base_text = ticket_phrase

        base_text = re.sub(r"\s+", " ", base_text).strip()
        base_text = _polish_description_text(base_text)
        instruction = (
            "Rewrite this as a concise Jira ticket description in professional American English. "
            "Respond with two sentences: explain the impact, then outline the next action. "
            "Return plain text only with no quotes, bullets, headings, or markdown."
        )
        return _rewrite_text(base_text, instruction=instruction)


ISSUE_TYPE_SYNONYMS: Dict[str, Set[str]] = {
    "Bug": {"bug", "bugs", "defect", "defects", "error", "issue", "fault", "bugfix", "buggy"},
    "Story": {"story", "stories", "feature", "feat", "improvement", "enhancement"},
    "Task": {"task", "todo", "chore", "work", "action", "item", "ticket", "job"},
    "Epic": {"epic", "big story", "initiative"},
}

PRIORITY_SYNONYMS: Dict[str, Set[str]] = {
    "Highest": {"blocker", "critical", "urgent", "p0", "sev0", "sev-0", "sev1", "highest"},
    "High": {"high", "p1", "sev2", "major", "important", "significant"},
    "Medium": {"medium", "p2", "normal", "regular", "standard", "default", "okay"},
    "Low": {"low", "minor", "p3", "p4", "lowest", "trivial", "nice", "optional", "later"},
}


def _build_vocab(synonyms: Dict[str, Set[str]]) -> Dict[str, str]:
    vocab: Dict[str, str] = {}
    for canonical, words in synonyms.items():
        for word in words:
            vocab[word] = canonical
    return vocab


ISSUE_TYPE_VOCAB = _build_vocab(ISSUE_TYPE_SYNONYMS)
PRIORITY_VOCAB = _build_vocab(PRIORITY_SYNONYMS)


ASSIGNEE_PATTERN = re.compile(
    r"assign(?:ed)?(?:\s+\w+){0,3}?\s+to\s+(?:@)?([a-zA-Z0-9._-]+(?:\s+[a-zA-Z0-9._-]+)?)",
    re.IGNORECASE,
)
ASSIGNEE_DIRECT_PATTERN = re.compile(
    r"\bassign\s+(?:it\s+to\s+|this\s+to\s+)?(?:@)?([a-zA-Z0-9._-]+(?:\s+[a-zA-Z0-9._-]+)?)",
    re.IGNORECASE,
)
TITLE_PATTERN = re.compile(
    r"\b(?:title|titled|summary)\s*[:=]?\s*(?:\"([^\"]+)\"|“([^”]+)”|'([^']+))",
    re.IGNORECASE,
)
DESCRIPTION_QUOTED_PATTERN = re.compile(
    r"\bdescription\s*[:=]?\s*(?:\"([^\"]+)\"|“([^”]+)”|'([^']+))",
    re.IGNORECASE | re.DOTALL,
)
DESCRIPTION_FALLBACK_PATTERN = re.compile(
    r"\bdescription\s*[:=]?\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)


def _normalize_tokens(message: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", message.lower())


def _match_vocabulary(
    message: str,
    vocab: Dict[str, str],
    default: Optional[str] = None,
    cutoff: float = 0.82,
) -> Optional[str]:
    lowered = message.lower()
    for word, canonical in vocab.items():
        if word in lowered:
            return canonical
    tokens = _normalize_tokens(message)
    if not tokens:
        return default
    words = list(vocab.keys())
    for token in tokens:
        if token in vocab:
            return vocab[token]
        match = difflib.get_close_matches(token, words, n=1, cutoff=cutoff)
        if match:
            return vocab[match[0]]
    return default


def _extract_issue_type(message: str) -> str:
    issue_type = _match_vocabulary(message, ISSUE_TYPE_VOCAB, default="Task")
    return issue_type or "Task"


def _extract_priority(message: str) -> Optional[str]:
    return _match_vocabulary(message, PRIORITY_VOCAB, default=None)


def _extract_assignee(message: str) -> Optional[str]:
    match = ASSIGNEE_PATTERN.search(message)
    if match:
        captured = match.group(1).strip()
        name = _sanitize_assignee_name(captured)
        if name:
            return name
    match = ASSIGNEE_DIRECT_PATTERN.search(message)
    if match:
        captured = match.group(1).strip()
        name = _sanitize_assignee_name(captured)
        if name:
            return name
    lower = message.lower()
    if re.search(r"assign(?:\s+this|\s+it|\s+the\s+ticket)?\s+to\s+me", lower):
        return "me"
    if re.search(r"assign\s+me\b", lower):
        return "me"
    if re.search(r"give\s+(?:me|it\s+to\s+me)", lower):
        return "me"
    return None


def _extract_labels(message: str) -> List[str]:
    labels = set()
    for hashtag in re.findall(r"#([a-zA-Z0-9_-]+)", message):
        labels.add(hashtag.lower())

    label_section = re.search(r"labels?:\s*([a-zA-Z0-9_,\s-]+)", message, re.IGNORECASE)
    if label_section:
        for part in label_section.group(1).split(","):
            cleaned = part.strip().lower()
            if cleaned:
                labels.add(cleaned.replace(" ", "-"))

    for tag_match in re.findall(r"tag(?:s)?\s+(?:are\s+)?([a-zA-Z0-9_-]+)", message, re.IGNORECASE):
        cleaned = tag_match.strip().lower()
        if cleaned:
            labels.add(cleaned.replace(" ", "-"))

    return sorted(labels)


def _extract_title(message: str) -> Optional[str]:
    match = TITLE_PATTERN.search(message)
    if match:
        for group in match.groups():
            if group:
                return group.strip()
    return None


def _extract_description(message: str, structured_hint: Optional[str] = None) -> Optional[str]:
    _, structured_description = _extract_structured_form_sections(message)
    if structured_hint:
        structured_description = structured_hint
    if structured_description:
        return re.sub(r"\s+", " ", structured_description).strip()

    match = DESCRIPTION_QUOTED_PATTERN.search(message)
    if match:
        for group in match.groups():
            if group:
                return re.sub(r"\s+", " ", group).strip()

    fallback = DESCRIPTION_FALLBACK_PATTERN.search(message)
    if fallback:
        desc = fallback.group(1).strip()
        desc = _strip_at_stop(desc)
        desc = re.sub(r"\s+", " ", desc).strip()
        return desc or None
    return None


def _strip_at_stop(text: str) -> str:
    if not text:
        return text
    pattern = re.compile(rf"^(.*?)(?:\b({'|'.join(STOP_TOKENS)})\b.*)$", re.IGNORECASE | re.DOTALL)
    match = pattern.match(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_ticket_phrase(message: str) -> Optional[str]:
    pattern = re.compile(
        (
            r"(?:make|create|open|file|raise|log|need|want|pls|please|do|build|setup|set up|fix)\s+"
            r"(?:an?\s+)?(?:bug|issue|ticket|story|task|item)?\s*(?:ticket|issue)?\s*(.+)"
        ),
        re.IGNORECASE,
    )
    match = pattern.search(message)
    if not match:
        return None
    phrase = match.group(1)
    phrase = _strip_at_stop(phrase)
    phrase = re.sub(r"\s+", " ", phrase).strip()
    if not phrase:
        return None
    if len(phrase.split()) < 3:
        return None
    return phrase


def _is_mostly_filler(text: str) -> bool:
    tokens = _normalize_tokens(text)
    if not tokens:
        return True
    meaningful = [token for token in tokens if token not in FILLER_TOKENS]
    return len(meaningful) == 0


def _strip_leading_fillers(text: str) -> str:
    if not text:
        return ""
    tokens = text.strip().split()
    idx = 0
    while idx < len(tokens) and tokens[idx].lower() in FILLER_TOKENS:
        idx += 1
    trimmed = " ".join(tokens[idx:])
    return trimmed if trimmed else text.strip()


def _sanitize_assignee_name(raw: str) -> Optional[str]:
    if not raw:
        return None
    tokens = raw.split()
    cleaned_tokens: List[str] = []
    for token in tokens:
        low = token.lower()
        if (
            low in STOP_TOKENS
            or low in FILLER_TOKENS
            or low
            in {
                "its",
                "it's",
                "urgent",
                "bug",
                "issue",
                "story",
                "task",
                "label",
                "labels",
                "tag",
                "tags",
            }
        ):
            break
        cleaned_tokens.append(token)
    name = " ".join(cleaned_tokens).strip()
    return name.lower() if name else None


def _remove_assignee_phrases(text: str, assignee_name: Optional[str]) -> str:
    if not text:
        return text

    cleaned = text
    if assignee_name:
        escaped_name = re.escape(assignee_name)
        pattern_with_to = re.compile(
            rf"\bassign(?:ed)?\s+(?:this|it|the\s+ticket)?\s+to\s+{escaped_name}\b",
            re.IGNORECASE,
        )
        pattern_direct = re.compile(
            rf"\bassign\s+{escaped_name}\b",
            re.IGNORECASE,
        )
        cleaned = pattern_with_to.sub("", cleaned)
        cleaned = pattern_direct.sub("", cleaned)

    cleaned = re.sub(r"\bassign\s+(?:me|myself)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\bassign(?:ed)?\s+(?:this|it|the\s+ticket)?\s+to\s+(?:me|myself)\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def _capitalize_sentence(text: str) -> str:
    if not text:
        return text
    text = text.strip()
    if not text:
        return text
    return text[0].upper() + text[1:]


def _polish_summary_text(text: str) -> str:
    cleaned = _strip_leading_fillers(text)
    if not cleaned:
        cleaned = text.strip()
    cleaned = cleaned.strip()
    if not cleaned:
        return "General issue"
    cleaned = _capitalize_sentence(cleaned)
    return cleaned[:120]


def _polish_description_text(text: str) -> str:
    cleaned = _strip_leading_fillers(text)
    if not cleaned:
        cleaned = text.strip()
    cleaned = cleaned.strip()
    if not cleaned:
        return "No additional details provided."
    cleaned = _capitalize_sentence(cleaned)
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _ensure_ollama_client() -> Optional[OllamaClient]:
    global _ollama_client
    if _ollama_client is not None:
        return _ollama_client
    if not (config.ENABLE_TEXT_REWRITE or config.ENABLE_INTENT_DETECTION):
        return None
    try:
        _ollama_client = OllamaClient(
            host=config.OLLAMA_HOST,
            model=config.OLLAMA_MODEL,
            timeout=config.OLLAMA_TIMEOUT,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("ollama_init_failed", exc_info=exc)
        _ollama_client = None
    return _ollama_client


def _rewrite_text(text: str, instruction: str, max_length: Optional[int] = None) -> str:
    if not text or not config.ENABLE_TEXT_REWRITE:
        return text
    client = _ensure_ollama_client()
    if not client:
        return text
    rewritten = client.rewrite(text, instruction=instruction)
    if not rewritten:
        return text
    rewritten = rewritten.strip()
    if max_length:
        rewritten = rewritten[:max_length]
    return rewritten


def parse_search_query(message: str) -> Optional[SearchQuery]:
    if not message or not message.strip():
        return None
    lowered = message.lower()
    issue_keys = _extract_issue_keys(message)
    looks_like_search = any(keyword in lowered for keyword in SEARCH_KEYWORDS)
    if not looks_like_search and not issue_keys:
        return None

    limit = _extract_limit(lowered)
    jql_parts: List[str] = []

    if issue_keys:
        key_list = ", ".join(issue_keys)
        jql_parts.append(f"issuekey in ({key_list})")

    terms = _extract_search_terms(message)
    if terms:
        escaped_terms = _escape_jql_term(terms)
        jql_parts.append(f'text ~ "{escaped_terms}"')

    if any(keyword in lowered for keyword in MY_KEYWORDS):
        jql_parts.append("assignee = currentUser()")
    if any(keyword in lowered for keyword in OPEN_KEYWORDS):
        jql_parts.append("statusCategory != Done")

    if not jql_parts:
        return None

    jql = " AND ".join(jql_parts) + " ORDER BY updated DESC"
    return SearchQuery(jql=jql, limit=limit, raw_terms=terms or None)


def _extract_search_terms(message: str) -> str:
    tokens = re.findall(r"[\w-]+", message)
    filtered: List[str] = []
    for token in tokens:
        lower = token.lower()
        if ISSUE_KEY_PATTERN.match(token):
            continue
        if lower in SEARCH_KEYWORDS or lower in SEARCH_STOPWORDS or lower in MY_KEYWORDS:
            continue
        if lower in OPEN_KEYWORDS or lower in FILLER_TOKENS:
            continue
        filtered.append(token)
    return " ".join(filtered).strip()


def _extract_issue_keys(message: str) -> List[str]:
    matches = ISSUE_KEY_PATTERN.findall(message)
    return [match.upper() for match in matches]


def _extract_limit(lowered_message: str, default: int = 5) -> int:
    match = re.search(r"\b(\d{1,2})\b", lowered_message)
    if not match:
        return default
    value = int(match.group(1))
    return max(1, min(value, 20))


def _escape_jql_term(term: str) -> str:
    return term.replace('"', '\\"')


def _should_use_rewrite(original: str, rewritten: str) -> bool:
    if not rewritten:
        return False

    original_tokens = {
        token for token in re.findall(r"[a-z0-9']+", original.lower()) if len(token) >= 4
    }
    if not original_tokens:
        return True

    rewritten_tokens = {
        token for token in re.findall(r"[a-z0-9']+", rewritten.lower()) if len(token) >= 4
    }
    if not rewritten_tokens:
        return False

    overlap = original_tokens & rewritten_tokens
    similarity = len(overlap) / len(original_tokens)
    return similarity >= 0.4


def _limit_summary_words(summary: str, max_words: int = 12) -> str:
    tokens = summary.split()
    if len(tokens) <= max_words:
        return summary
    truncated = " ".join(tokens[:max_words])
    return truncated.rstrip(".,;:")


def _extract_structured_form_sections(message: str) -> tuple[Optional[str], Optional[str]]:
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    if len(lines) <= 1:
        return None, None

    summary_candidate: Optional[str] = None
    fallback_candidate: Optional[str] = None
    description_lines: List[str] = []
    capturing_description = False

    for line in lines:
        lower = line.lower()
        if lower in FORM_SECTION_HEADERS:
            capturing_description = lower == "description"
            continue
        if lower in FORM_SECTION_SKIP:
            capturing_description = False
            continue
        if lower in STATUS_KEYWORDS:
            capturing_description = False
            continue
        if ISSUE_KEY_PATTERN.match(line):
            capturing_description = False
            continue
        if re.fullmatch(r"\d+", line):
            capturing_description = False
            continue

        if capturing_description:
            description_lines.append(line)
            continue

        if not summary_candidate:
            token_count = len(line.split())
            if (
                token_count >= 3
                and lower not in FORM_SECTION_HEADERS
                and lower not in FORM_SECTION_SKIP
            ):
                summary_candidate = line
            elif (
                not fallback_candidate
                and lower not in FORM_SECTION_HEADERS
                and lower not in FORM_SECTION_SKIP
            ):
                fallback_candidate = line

    if not summary_candidate:
        summary_candidate = fallback_candidate

    description_text = " ".join(description_lines).strip()

    if summary_candidate and len(summary_candidate.split()) < 3 and description_lines:
        summary_candidate = description_lines[0]

    return summary_candidate, (description_text or None)


def _extract_summary(
    message: str,
    explicit_title: Optional[str] = None,
    structured_hint: Optional[str] = None,
) -> str:
    structured_summary, _ = _extract_structured_form_sections(message)
    if structured_hint:
        structured_summary = structured_hint
    if explicit_title:
        text = explicit_title
        allow_rewrite = config.ENABLE_TEXT_REWRITE and not structured_hint
    elif structured_summary:
        text = structured_summary
        allow_rewrite = False  # structured forms already provide concise summary
    else:
        phrase = _extract_ticket_phrase(message)
        text = phrase or message.strip()
        allow_rewrite = config.ENABLE_TEXT_REWRITE
    if not text:
        return ""
    text = re.sub(r"[\r\n]+", " ", text)
    sentence_end = re.split(r"(?<=[.!?])\s+", text)
    summary = sentence_end[0]
    summary = re.sub(r"\s+", " ", summary)
    summary = _polish_summary_text(summary)

    rewrite_allowed = allow_rewrite or (
        structured_hint is not None and config.ENABLE_TEXT_REWRITE and len(summary.split()) > 12
    )

    if rewrite_allowed:
        rewritten = _rewrite_text(
            summary,
            instruction=(
                "Rewrite this as a concise Jira ticket summary in professional American English. "
                "Use an imperative verb, stay under 12 words, and respond with the sentence only. "
                "Do not add quotes, numbering, or markdown."
            ),
            max_length=120,
        )
        if rewritten and _should_use_rewrite(summary, rewritten):
            summary = rewritten

    return _limit_summary_words(summary)


def _normalize_common_typos(text: str) -> str:
    replacements = {
        "assing": "assign",
        "asign": "assign",
    }
    normalized = text
    for wrong, right in replacements.items():
        normalized = re.sub(rf"\b{wrong}\b", right, normalized, flags=re.IGNORECASE)
    return normalized


def parse_ticket_request(
    message: str, assignee_map: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Parse a natural language message into Jira ticket fields."""

    ticket = TicketParser(message, assignee_map).parse()
    return ticket.to_dict()
