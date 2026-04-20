"""
exceptions.py — Custom exception hierarchy for WorkMind AI
============================================================
Provides typed, descriptive exceptions for every failure mode in the
application, replacing bare ``Exception`` raises with domain-specific
errors that carry structured context for logging and error reporting.
"""

from __future__ import annotations


class WorkMindError(Exception):
    """Base exception for all WorkMind AI errors."""


class APIKeyMissingError(WorkMindError):
    """Raised when no GEMINI_API_KEY is found in env or secrets."""

    def __init__(self) -> None:
        super().__init__(
            "GEMINI_API_KEY not found. Set it via environment variable "
            "or .streamlit/secrets.toml."
        )


class ModelNotFoundError(WorkMindError):
    """Raised when no available Gemini model can serve the request."""

    def __init__(self, agent_name: str, last_error: str) -> None:
        self.agent_name = agent_name
        self.last_error = last_error
        super().__init__(
            f"[{agent_name}] No available Gemini model found. "
            f"Last error: {last_error}"
        )


class QuotaExceededError(WorkMindError):
    """Raised when the Gemini API quota is exhausted after all retries."""

    def __init__(self, agent_name: str, retries: int) -> None:
        self.agent_name = agent_name
        self.retries = retries
        super().__init__(
            f"[{agent_name}] API quota exceeded after {retries} retries. "
            "Please wait ~60 seconds and try again."
        )


class InputValidationError(WorkMindError):
    """Raised when user input fails validation rules."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class PromptInjectionError(WorkMindError):
    """Raised when a potential prompt-injection attempt is detected."""

    def __init__(self) -> None:
        super().__init__(
            "Potential prompt injection detected. "
            "Your input contains phrases that may attempt to override "
            "the agent's instructions. Please rephrase your request."
        )


class JSONParseError(WorkMindError):
    """Raised when the Gemini response cannot be parsed as valid JSON."""

    def __init__(self, raw_response: str) -> None:
        self.raw_response = raw_response[:200]  # Truncate for safety
        super().__init__(
            "Failed to parse structured JSON from the AI response. "
            "The agent may have returned an unexpected format."
        )
