# llm_client.py
"""Utilities for interacting with local LLM providers (Ollama)."""

from __future__ import annotations

import logging
from typing import Any, Optional, cast

import requests  # type: ignore[import-untyped]


class OllamaClient:
    """Simple HTTP client for the Ollama REST API."""

    def __init__(self, host: str, model: str, timeout: int = 30) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._endpoint = f"{self.host}/api/generate"

    def rewrite(self, text: str, instruction: str) -> Optional[str]:
        """Rewrite the given text according to the instruction.

        Returns the rewritten text, or ``None`` if the rewrite fails.
        """
        payload = {
            "model": self.model,
            "prompt": self._build_prompt(text, instruction),
            "stream": False,
        }
        try:
            response = requests.post(self._endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            logging.warning("Ollama request failed: %s", exc)
            return None

        data: dict[str, Any] = response.json()
        rewritten = cast(Optional[str], data.get("response"))
        if not rewritten:
            logging.warning("Ollama returned no response for rewrite request.")
            return None
        return rewritten.strip()

    def detect_intent(self, text: str) -> Optional[str]:
        """Classify the intent of the message using the LLM."""

        prompt = (
            "You are an intent classifier for a Jira ticket assistant. "
            "Classify the user's message into one of the following labels: "
            "greeting, ticket_request, other. Respond with only the label in lowercase.\n\n"
            f"Message: {text}\n"
        )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        try:
            response = requests.post(self._endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            logging.warning("Ollama intent detection failed: %s", exc)
            return None

        data: dict[str, Any] = response.json()
        label = cast(Optional[str], data.get("response"))
        if not label:
            return None
        return label.strip().lower()

    @staticmethod
    def _build_prompt(text: str, instruction: str) -> str:
        return (
            "You are an assistant that rewrites text in concise, professional American "
            "English suitable for Jira tickets. Follow the instruction exactly and reply "
            "with only the rewritten text.\n\n"
            f"Instruction: {instruction}\n\n"
            "Text to rewrite:"
            f"\n{text}\n"
        )
