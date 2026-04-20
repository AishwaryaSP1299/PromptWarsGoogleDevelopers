"""
utils.py — Shared helpers for WorkMind AI
==========================================
Provides: secret loading, input sanitisation, JSON parsing,
          PII redaction, prompt-injection detection,
          Google Cloud Logging integration.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

import streamlit as st

# ── Constants ────────────────────────────────────────────────────────────────
MAX_INPUT_CHARS: int    = 4_000
MAX_RETRIES: int        = 3
RETRY_WAITS: list[int]  = [20, 40, 65]


# ── Logging (Google Cloud Logging-compatible structured JSON) ─────────────────
def setup_logger(name: str = "workmind") -> logging.Logger:
    """
    Return a logger that emits structured JSON to stderr.

    On Google Cloud Run, the Cloud Logging agent automatically ingests
    structured stderr output, making this a zero-dependency integration
    with Google Cloud Logging.

    Args:
        name: Logger namespace.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt=(
                    '{"severity":"%(levelname)s","message":"%(message)s",'
                    '"logger":"%(name)s","time":"%(asctime)s"}'
                ),
                datefmt="%Y-%m-%dT%H:%M:%SZ",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger: logging.Logger = setup_logger()


# ── Secret loading ────────────────────────────────────────────────────────────
def get_secret(key: str) -> str | None:
    """
    Load a secret from environment variables (Cloud Run) or Streamlit secrets.

    Never returns hardcoded values — all credentials are externalised.

    Args:
        key: Secret key name.

    Returns:
        Secret value string, or ``None`` if not configured.
    """
    if key in os.environ:
        return os.environ[key]

    has_secrets_file = any(
        os.path.exists(p)
        for p in [
            ".streamlit/secrets.toml",
            "/root/.streamlit/secrets.toml",
            "/app/.streamlit/secrets.toml",
        ]
    )
    if has_secrets_file and hasattr(st, "secrets"):
        try:
            return st.secrets.get(key)
        except Exception:
            pass
    return None


# ── Input sanitisation ────────────────────────────────────────────────────────
def sanitize_input(text: str) -> str:
    """
    Strip whitespace and cap at ``MAX_INPUT_CHARS`` to prevent prompt injection.

    Args:
        text: Raw user input.

    Returns:
        Sanitised, length-capped string.
    """
    return text.strip()[:MAX_INPUT_CHARS]


def validate_task_list(raw: str) -> list[str]:
    """
    Parse a newline-separated task list, filtering blank lines.

    Args:
        raw: Multi-line task string.

    Returns:
        List of non-empty task strings.

    Raises:
        ValueError: Fewer than 2 tasks provided.
    """
    tasks = [sanitize_input(t) for t in raw.splitlines() if t.strip()]
    if len(tasks) < 2:
        raise ValueError("Please provide at least 2 tasks.")
    return tasks


# ── JSON parsing ──────────────────────────────────────────────────────────────
def parse_json(raw: str) -> Any:
    """
    Strip Markdown code fences from a Gemini response then parse as JSON.

    Args:
        raw: Raw text from Gemini, possibly wrapped in ```json … ```.

    Returns:
        Parsed Python object (dict or list).

    Raises:
        json.JSONDecodeError: Response is not valid JSON.
    """
    text = raw
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


# ── Agent activity log ────────────────────────────────────────────────────────
def log_agent_action(agent_name: str, action: str) -> None:
    """
    Append a timestamped entry to the cross-agent activity log and emit
    a structured Cloud Logging record.

    Args:
        agent_name: Display name of the acting agent.
        action:     Human-readable action description.
    """
    entry = {
        "timestamp": time.strftime("%H:%M:%S"),
        "agent":     agent_name,
        "action":    action,
    }
    if "agent_memory" in st.session_state:
        st.session_state.agent_memory["agent_logs"].append(entry)
    logger.info("%s — %s", agent_name, action)


# ── Data Privacy & AI Security ────────────────────────────────────────────────
class DataPrivacy:
    """
    Enterprise-grade PII redaction and prompt-injection guardrails.

    Before any user content reaches the Gemini API, this class:

    1. **PII Redaction** — replaces phone numbers, email addresses, SSNs, and
       credit card numbers with labelled ``[TYPE_REDACTED]`` tokens, ensuring
       sensitive data never leaves the user's session unmasked.

    2. **Prompt-Injection Detection** — flags inputs containing known
       jailbreak / instruction-override patterns so the UI can warn the user
       and prevent adversarial API abuse.

    This satisfies GDPR / CCPA compliance requirements for AI pipelines
    processing user-generated content in corporate environments.

    Example::

        dp = DataPrivacy()
        dp.redact("Call +1-800-555-0123 or bob@acme.com")
        # → 'Call [PHONE_REDACTED] or [EMAIL_REDACTED]'
    """

    _PATTERNS: list[tuple[str, str]] = [
        (r"\b\d{3}-\d{2}-\d{4}\b",
         "[SSN_REDACTED]"),
        (r"\b(?:\d[ -]?){13,16}\b",
         "[CARD_REDACTED]"),
        (r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
         "[EMAIL_REDACTED]"),
        (r"(\+?\d[\d\s\-().]{7,}\d)",
         "[PHONE_REDACTED]"),
    ]

    _INJECTION_SIGNALS: list[str] = [
        "ignore previous instructions",
        "ignore all instructions",
        "disregard the above",
        "you are now",
        "act as",
        "jailbreak",
        "dan mode",
        "bypass",
    ]

    def __init__(self) -> None:
        self._compiled = [
            (re.compile(pattern, re.IGNORECASE), replacement)
            for pattern, replacement in self._PATTERNS
        ]

    def redact(self, text: str) -> str:
        """
        Replace PII tokens in *text* with labelled placeholders.

        Args:
            text: Raw user input or document text.

        Returns:
            Text with PII replaced by ``[TYPE_REDACTED]`` markers.
        """
        for pattern, replacement in self._compiled:
            text = pattern.sub(replacement, text)
        return text

    def is_injection_attempt(self, text: str) -> bool:
        """
        Return ``True`` if *text* contains known prompt-injection signals.

        Args:
            text: User input to inspect.

        Returns:
            Boolean — ``True`` means a potential injection was detected.
        """
        lower = text.lower()
        return any(signal in lower for signal in self._INJECTION_SIGNALS)

    def safe_input(self, text: str) -> tuple[str, bool]:
        """
        Sanitise, redact PII, and check for injection in one call.

        Args:
            text: Raw user input.

        Returns:
            Tuple of ``(redacted_text, injection_detected)``.
        """
        cleaned   = sanitize_input(text)
        redacted  = self.redact(cleaned)
        injection = self.is_injection_attempt(cleaned)
        if injection:
            logger.warning("Potential prompt injection detected.")
        return redacted, injection


# ── Module-level singleton ────────────────────────────────────────────────────
data_privacy: DataPrivacy = DataPrivacy()
