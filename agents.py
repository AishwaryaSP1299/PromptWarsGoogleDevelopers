"""
agents.py — AI Agent definitions for WorkMind AI
==================================================
Each agent encapsulates a distinct productivity persona, a dedicated
system prompt, and the Gemini API call logic with quota-aware retry.

Architecture:
    Agent (base class)
    ├── task_agent         — Eisenhower Matrix Specialist
    ├── meeting_agent      — Meeting Intelligence Specialist
    ├── email_agent        — Business Communication Specialist
    └── planner_agent      — Schedule Optimisation Specialist

    AgentOrchestrator      — Concurrent multi-agent dispatcher
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import google.generativeai as genai
import streamlit as st

from exceptions import ModelNotFoundError, QuotaExceededError
from utils import MAX_RETRIES, RETRY_WAITS, log_agent_action, sanitize_input


class Agent:
    """
    Specialised AI agent with a unique persona and system prompt.

    Each agent prepends its ``system_prompt`` to every user prompt before
    calling the Gemini API, ensuring a consistent, distinct identity.

    Attributes:
        name:          Display name shown in the UI and activity log.
        emoji:         Emoji icon for visual identification.
        role:          Short role descriptor shown in the agent header.
        system_prompt: Persona instruction prepended to every API call.
        color:         Hex colour used for UI accents.
    """

    __slots__ = ("name", "emoji", "role", "system_prompt", "color")

    def __init__(
        self,
        name: str,
        emoji: str,
        role: str,
        system_prompt: str,
        color: str,
    ) -> None:
        self.name          = name
        self.emoji         = emoji
        self.role          = role
        self.system_prompt = system_prompt
        self.color         = color

    def __repr__(self) -> str:
        return f"Agent(name={self.name!r}, role={self.role!r})"

    # ── Gemini API call ───────────────────────────────────────────────────────
    def call(
        self,
        user_prompt: str,
        available_models: list[str],
        max_retries: int = MAX_RETRIES,
    ) -> str:
        """
        Send ``user_prompt`` to Gemini, prepended by this agent's system prompt.

        Implements:
        - **Model fallback**: iterates ``available_models`` on 404 errors.
        - **Quota backoff**: waits ``RETRY_WAITS[attempt]`` seconds on 429/503.

        Args:
            user_prompt:       The task-specific prompt from the UI.
            available_models:  Ordered list of model names to try.
            max_retries:       Maximum quota-retry attempts before raising.

        Returns:
            The model's text response, stripped of leading/trailing whitespace.

        Raises:
            ModelNotFoundError: Every available model returned 404.
            QuotaExceededError: Rate-limit retries exhausted.
        """
        full_prompt = f"{self.system_prompt}\n\n{sanitize_input(user_prompt)}"

        for attempt in range(max_retries + 1):
            last_404: str | None = None

            for model_name in available_models:
                try:
                    model    = genai.GenerativeModel(model_name)
                    response = model.generate_content(full_prompt)
                    log_agent_action(self.name, f"Response via {model_name}")
                    return response.text.strip()

                except Exception as exc:
                    err = str(exc)
                    if "404" in err or "not found" in err.lower():
                        last_404 = err
                        continue
                    if any(c in err for c in ("429", "TooManyRequests", "503", "quota")):
                        break
                    raise
            else:
                if last_404:
                    raise ModelNotFoundError(self.name, last_404)

            if attempt < max_retries:
                wait = RETRY_WAITS[attempt]
                placeholder = st.empty()
                for remaining in range(wait, 0, -1):
                    placeholder.warning(
                        f"⏳ **{self.name}** — quota limit. "
                        f"Retrying in {remaining}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(1)
                placeholder.empty()
            else:
                raise QuotaExceededError(self.name, max_retries)

    # ── UI rendering ──────────────────────────────────────────────────────────
    def render_header(self, status: str = "ONLINE") -> None:
        """Render a styled, accessible agent identity header."""
        st.markdown(
            f"""
            <div role="banner" aria-label="{self.name} agent status"
                 style="display:flex;align-items:center;gap:10px;margin-bottom:0.3rem">
                <span aria-hidden="true" style="font-size:1.4rem">{self.emoji}</span>
                <span style="font-size:0.8rem;font-weight:700;letter-spacing:0.06em;
                             text-transform:uppercase;color:#89b4fa">{self.name}</span>
                <span role="status" aria-label="Agent is {status}"
                      style="font-size:0.72rem;padding:2px 8px;border-radius:20px;
                             background:#21c45422;color:#21c454;border:1px solid #21c45455">
                    ● {status}
                </span>
                <span style="font-size:0.78rem;color:#6c7086">{self.role}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Agent instances ───────────────────────────────────────────────────────────
task_agent = Agent(
    name="TASK-PRIORITIZER AGENT",
    emoji="🗂️",
    role="Eisenhower Matrix Specialist",
    color="#f59e0b",
    system_prompt=(
        "You are the Task Prioritizer Agent — a productivity expert specialising "
        "in the Eisenhower Matrix (Do First / Schedule / Delegate / Drop). "
        "Analyse tasks through the lens of urgency and impact. "
        "Respond ONLY with a valid JSON array. No preamble, no markdown."
    ),
)

meeting_agent = Agent(
    name="MEETING-SUMMARIZER AGENT",
    emoji="📝",
    role="Meeting Intelligence Specialist",
    color="#0ea5e9",
    system_prompt=(
        "You are the Meeting Summarizer Agent — an expert facilitator. "
        "Extract: executive summary, key decisions, action items (owners + deadlines), "
        "and open questions from raw meeting notes. "
        "Respond ONLY with a valid JSON object. No preamble, no markdown."
    ),
)

email_agent = Agent(
    name="EMAIL-DRAFTER AGENT",
    emoji="✉️",
    role="Business Communication Specialist",
    color="#a855f7",
    system_prompt=(
        "You are the Email Drafter Agent — a world-class business communication expert. "
        "Write professional, tone-aware emails. When meeting context is provided, "
        "use it to write specific, informed emails. "
        "Respond ONLY with JSON: {'subject':'...','body':'...'}."
    ),
)

planner_agent = Agent(
    name="DAY-PLANNER AGENT",
    emoji="📅",
    role="Schedule Optimisation Specialist",
    color="#10b981",
    system_prompt=(
        "You are the Day Planner Agent — a productivity coach using cognitive load theory. "
        "Schedule Deep Work during peak energy windows; cluster shallow tasks in low-energy slots. "
        "Respond ONLY with a valid JSON object. No preamble, no markdown."
    ),
)

ALL_AGENTS: tuple[Agent, ...] = (task_agent, meeting_agent, email_agent, planner_agent)


# ── Agent Orchestrator ────────────────────────────────────────────────────────
class AgentOrchestrator:
    """
    Concurrent multi-agent dispatcher using ``ThreadPoolExecutor``.

    Fans out independent agent calls concurrently, reducing end-to-end
    latency for pipelines that invoke multiple agents on the same input.

    Args:
        available_models: Ordered list of Gemini model names to use.

    Example::

        orch = AgentOrchestrator(available_models=AVAILABLE_MODELS)
        results = orch.run_parallel([
            (task_agent,    task_prompt),
            (meeting_agent, notes_prompt),
        ])
    """

    __slots__ = ("available_models",)

    def __init__(self, available_models: list[str]) -> None:
        self.available_models = available_models

    def run_parallel(
        self,
        jobs: list[tuple[Agent, str]],
        max_workers: int = 4,
    ) -> dict[str, str]:
        """
        Execute multiple agent calls concurrently.

        Args:
            jobs:        List of ``(Agent, prompt_str)`` pairs.
            max_workers: Thread pool size.

        Returns:
            Dict mapping ``agent.name → response_text``.
            Failed agents have values prefixed with ``"ERROR: "``.
        """
        results: dict[str, str] = {}

        def _call(agent: Agent, prompt: str) -> tuple[str, str]:
            try:
                return agent.name, agent.call(prompt, self.available_models)
            except Exception as exc:
                log_agent_action(agent.name, f"ERROR: {exc}")
                return agent.name, f"ERROR: {exc}"

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_call, agent, prompt): agent
                for agent, prompt in jobs
            }
            for future in as_completed(futures):
                name, result = future.result()
                results[name] = result
                log_agent_action(name, "Parallel execution completed")

        return results
