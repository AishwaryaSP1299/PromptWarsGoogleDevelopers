"""
conftest.py — pytest configuration and shared fixtures for WorkMind AI tests.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def gemini_api_key(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Return a placeholder API key for unit tests (no real API calls made)."""
    return "TEST_KEY_PLACEHOLDER"


@pytest.fixture()
def mock_agent_memory() -> dict:
    """Return a clean agent memory dict matching app.py session state schema."""
    return {
        "last_meeting_summary": None,
        "last_task_list":       None,
        "last_email_draft":     None,
        "last_schedule":        None,
        "agent_logs":           [],
    }
