"""
app.py — WorkMind AI: Intelligent Productivity Assistant
=========================================================
Vertical  : Productivity / Work
Stack     : Python · Streamlit · Google Gemini API · Google Cloud Run
GitHub    : github.com/AishwaryaSP1299

Entry point for the Streamlit application.  Business logic lives in:
  - agents.py  (Agent class + 4 agent instances)
  - utils.py   (secrets, sanitisation, JSON parsing, logging)
"""

from __future__ import annotations

import json
import os
from datetime import date

import google.generativeai as genai
import streamlit as st

from agents import email_agent, meeting_agent, planner_agent, task_agent
from utils import (
    get_secret,
    log_agent_action,
    parse_json,
    sanitize_input,
    validate_task_list,
)

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WorkMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.result-card {
    background: #1e1e2e; border: 1px solid #313244;
    border-radius: 12px; padding: 1.2rem 1.5rem; margin-top: 1rem;
}
.context-banner {
    background: #7c3aed22; border: 1px solid #7c3aed55;
    border-radius: 10px; padding: 0.7rem 1rem;
    margin-bottom: 1rem; font-size: 0.85rem; color: #c4b5fd;
}
.badge {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600; margin-right: 6px;
}
.badge-high   { background:#ff4b4b22; color:#ff4b4b; border:1px solid #ff4b4b55; }
.badge-medium { background:#ffa50022; color:#ffa500; border:1px solid #ffa50055; }
.badge-low    { background:#21c45422; color:#21c454; border:1px solid #21c45455; }
</style>
""", unsafe_allow_html=True)

# ── Session state (shared agent memory) ──────────────────────────────────────
if "agent_memory" not in st.session_state:
    st.session_state.agent_memory: dict = {
        "last_meeting_summary": None,
        "last_task_list":       None,
        "last_email_draft":     None,
        "last_schedule":        None,
        "agent_logs":           [],
    }

# ── Gemini API setup ──────────────────────────────────────────────────────────
GEMINI_API_KEY: str | None = get_secret("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error(
        "🔑 **GEMINI_API_KEY not found.**  "
        "Pass it via `--set-env-vars GEMINI_API_KEY=…` on Cloud Run."
    )
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)


@st.cache_resource(show_spinner="🔍 Detecting available Gemini models…")
def get_available_models() -> list[str]:
    """
    Query the Gemini API to discover which models are accessible for this
    API key and support ``generateContent``.  Results are cached for the
    lifetime of the Streamlit server process.

    Returns:
        Ordered list of model names, flash models first.
    """
    try:
        available = [
            m.name
            for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]
        available.sort(key=lambda n: (0 if "flash" in n else 1 if "pro" in n else 2))
        return available
    except Exception as exc:
        st.error(f"Could not list Gemini models: {exc}")
        return []


AVAILABLE_MODELS: list[str] = get_available_models()

if not AVAILABLE_MODELS:
    st.error("No usable Gemini models found for your API key.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 WorkMind AI")
    st.caption("Multi-Agent Productivity Suite")
    st.markdown("---")
    tool: str = st.radio(
        label="Select Agent",
        options=[
            "🗂️  Task Prioritizer",
            "📝  Meeting Summarizer",
            "✉️  Email Drafter",
            "📅  Day Planner",
        ],
        label_visibility="collapsed",
        help="Choose an agent to activate.",
    )
    st.markdown("---")
    st.markdown("#### 🔗 Agent Activity Log")
    logs: list[dict] = st.session_state.agent_memory.get("agent_logs", [])
    if logs:
        for entry in reversed(logs[-6:]):
            st.caption(
                f"`{entry['timestamp']}` **{entry['agent']}** — {entry['action']}"
            )
    else:
        st.caption("No agent activity yet.")
    st.markdown("---")
    st.caption(f"📆 {date.today().strftime('%A, %d %B %Y')}")
    st.caption("Powered by Google Gemini API")


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 1 — TASK PRIORITIZER
# ─────────────────────────────────────────────────────────────────────────────
if tool == "🗂️  Task Prioritizer":
    task_agent.render_header()
    st.markdown(
        "Submit your task list and the **Task Prioritizer Agent** will rank them "
        "by urgency × impact using the Eisenhower Matrix."
    )

    tasks_input: str = st.text_area(
        label="Your tasks — one per line",
        placeholder="Respond to client email\nFix production bug\nPrepare weekly report",
        height=180,
        help="Enter each task on a new line. Minimum 2 tasks required.",
    )
    context_input: str = st.text_input(
        label="Context for the agent (optional)",
        placeholder="e.g. Sprint ends tomorrow, client is a key account",
        help="Additional context helps the agent make better priority decisions.",
    )

    if st.button(
        "▶ Run Task Prioritizer Agent",
        type="primary",
        disabled=not tasks_input.strip(),
    ):
        try:
            tasks = validate_task_list(tasks_input)
        except ValueError as exc:
            st.warning(str(exc))
            st.stop()

        with st.spinner("🗂️ Task Prioritizer Agent is analysing your tasks…"):
            prompt = (
                "Analyse and prioritise the following tasks.\n"
                "For each task assign:\n"
                '- "priority": "High", "Medium", or "Low"\n'
                '- "quadrant": one of "Do First", "Schedule", "Delegate", "Drop"\n'
                '- "reason": one concise sentence explaining the rating\n\n'
                f"Context: {sanitize_input(context_input) or 'None'}\n\n"
                "Tasks:\n"
                + "\n".join(f"- {t}" for t in tasks)
                + '\n\nRespond ONLY with a valid JSON array:\n'
                '[{"task":"...","priority":"High","quadrant":"Do First","reason":"..."}]'
            )
            try:
                results = parse_json(task_agent.call(prompt, AVAILABLE_MODELS))
                st.session_state.agent_memory["last_task_list"] = results
                log_agent_action("TASK-PRIORITIZER AGENT", f"Prioritised {len(results)} tasks")

                st.subheader("Prioritised Task List")
                for item in results:
                    p   = item.get("priority", "Medium")
                    cls = {"High": "badge-high", "Medium": "badge-medium",
                           "Low": "badge-low"}.get(p, "badge-medium")
                    st.markdown(
                        f'<div class="result-card">'
                        f'<span class="badge {cls}">{p}</span>'
                        f'<span class="badge badge-medium">{item.get("quadrant","")}</span>'
                        f'<strong style="font-size:1rem">&nbsp;{item.get("task","")}</strong>'
                        f'<p style="color:#aaa;margin:.5rem 0 0;font-size:.9rem">'
                        f'💡 {item.get("reason","")}</p></div>',
                        unsafe_allow_html=True,
                    )
                st.success("✅ Results saved — the Day Planner Agent can import these tasks.")
            except Exception as exc:
                st.error(f"Agent failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 2 — MEETING SUMMARIZER
# ─────────────────────────────────────────────────────────────────────────────
elif tool == "📝  Meeting Summarizer":
    meeting_agent.render_header()
    st.markdown(
        "Paste raw meeting notes and the **Meeting Summarizer Agent** will extract "
        "a structured summary, decisions, action items, and open questions. "
        "The output is automatically shared with the Email Drafter Agent."
    )

    meeting_title: str = st.text_input(
        label="Meeting title (optional)",
        placeholder="e.g. Q2 Planning Sync",
        help="Used as a heading in the output and in the Email Drafter context.",
    )
    notes_input: str = st.text_area(
        label="Paste your meeting notes here",
        height=260,
        help="Raw, unstructured notes work best. Minimum 3-4 sentences recommended.",
    )

    if st.button(
        "▶ Run Meeting Summarizer Agent",
        type="primary",
        disabled=not notes_input.strip(),
    ):
        with st.spinner("📝 Meeting Summarizer Agent is processing…"):
            prompt = (
                f"Meeting: {sanitize_input(meeting_title) or 'Untitled'}\n"
                f"Notes:\n{sanitize_input(notes_input)}\n\n"
                "Respond ONLY with valid JSON:\n"
                '{"summary":"2-3 sentence overview",'
                '"decisions":["..."],'
                '"action_items":[{"owner":"Name or TBD","task":"...","due":"TBD"}],'
                '"open_questions":["..."]}'
            )
            try:
                data = parse_json(meeting_agent.call(prompt, AVAILABLE_MODELS))
                st.session_state.agent_memory["last_meeting_summary"] = {
                    "title": meeting_title or "Untitled Meeting",
                    "data":  data,
                }
                log_agent_action(
                    "MEETING-SUMMARIZER AGENT",
                    f"Summarised '{meeting_title or 'Untitled'}' — context ready for Email Drafter",
                )

                st.subheader(f"📋 {meeting_title or 'Meeting'} — Summary")
                st.info(data.get("summary", "—"))

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### ✅ Decisions")
                    for d in data.get("decisions", []) or ["None recorded."]:
                        st.markdown(f"- {d}")
                    st.markdown("#### ❓ Open Questions")
                    for q in data.get("open_questions", []) or ["None recorded."]:
                        st.markdown(f"- {q}")
                with col2:
                    st.markdown("#### 🎯 Action Items")
                    for item in data.get("action_items", []):
                        st.markdown(
                            f'<div class="result-card">'
                            f'<strong>{item.get("task","")}</strong><br>'
                            f'<span style="color:#aaa;font-size:.85rem">'
                            f'👤 {item.get("owner","TBD")} &nbsp;|&nbsp;'
                            f'📅 {item.get("due","TBD")}</span></div>',
                            unsafe_allow_html=True,
                        )
                st.success("✅ Summary saved — switch to **Email Drafter** to use it!")
            except Exception as exc:
                st.error(f"Agent failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 3 — EMAIL DRAFTER
# ─────────────────────────────────────────────────────────────────────────────
elif tool == "✉️  Email Drafter":
    email_agent.render_header()
    st.markdown(
        "Describe your intent and the **Email Drafter Agent** writes a "
        "professional, context-aware email instantly."
    )

    meeting_ctx: dict | None = st.session_state.agent_memory.get("last_meeting_summary")
    use_meeting_ctx = False
    if meeting_ctx:
        st.markdown(
            f'<div class="context-banner">🔗 <strong>Context from Meeting Summarizer Agent</strong>'
            f'<br>Meeting: <em>{meeting_ctx["title"]}</em> — '
            f'{meeting_ctx["data"].get("summary","")[:120]}…</div>',
            unsafe_allow_html=True,
        )
        use_meeting_ctx = st.checkbox(
            "✅ Inject meeting summary as context for this email",
            value=True,
            help="When enabled, the agent uses meeting decisions and action items to write a more specific email.",
        )

    col1, col2 = st.columns(2)
    with col1:
        recipient: str = st.text_input(
            "Recipient",
            placeholder="e.g. Client at Acme Corp",
            help="Who will receive this email?",
        )
        tone: str = st.selectbox(
            "Tone",
            ["Professional", "Friendly", "Assertive", "Apologetic", "Concise"],
            help="Sets the overall communication style.",
        )
    with col2:
        subject_hint: str = st.text_input(
            "Subject hint (optional)",
            placeholder="e.g. Project delay update",
        )
        length: str = st.selectbox(
            "Length",
            ["Short (3–4 sentences)", "Medium (2 paragraphs)", "Detailed (3+ paragraphs)"],
        )

    intent: str = st.text_area(
        "What do you want to communicate?",
        placeholder="e.g. Follow up on the action items from last sprint retro.",
        height=120,
        help="Plain English description of the email's purpose.",
    )

    if st.button(
        "▶ Run Email Drafter Agent",
        type="primary",
        disabled=not intent.strip(),
    ):
        with st.spinner("✉️ Email Drafter Agent is writing…"):
            ctx_block = ""
            if use_meeting_ctx and meeting_ctx:
                m = meeting_ctx["data"]
                ctx_block = (
                    f"\nCONTEXT FROM MEETING SUMMARIZER AGENT:\n"
                    f"Meeting: {meeting_ctx['title']}\n"
                    f"Summary: {m.get('summary','')}\n"
                    f"Decisions: {', '.join(m.get('decisions',[]))}\n"
                    f"Action Items: {json.dumps(m.get('action_items',[]))}\n"
                )

            prompt = (
                f"Draft a {tone.lower()} email:\n"
                f"- Recipient: {sanitize_input(recipient) or 'the recipient'}\n"
                f"- Subject hint: {sanitize_input(subject_hint) or 'derive from context'}\n"
                f"- Length: {length}\n"
                f"- Intent: {sanitize_input(intent)}\n"
                f"{ctx_block}\n"
                'Respond ONLY with valid JSON:\n'
                '{"subject":"Email subject line","body":"Full email body"}'
            )
            try:
                data = parse_json(email_agent.call(prompt, AVAILABLE_MODELS))
                st.session_state.agent_memory["last_email_draft"] = data
                log_agent_action("EMAIL-DRAFTER AGENT", f"Drafted: '{data.get('subject','')}'")

                st.subheader("Your Drafted Email")
                st.text_input("Subject", value=data.get("subject", ""))
                st.text_area("Body", value=data.get("body", ""), height=320)
                st.success("✅ Email drafted. Copy it from the fields above.")
            except Exception as exc:
                st.error(f"Agent failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 4 — DAY PLANNER
# ─────────────────────────────────────────────────────────────────────────────
elif tool == "📅  Day Planner":
    planner_agent.render_header()
    st.markdown(
        "Provide your schedule constraints and the **Day Planner Agent** "
        "builds an optimised, time-blocked day aligned to your energy levels."
    )

    task_ctx: list | None = st.session_state.agent_memory.get("last_task_list")
    use_task_ctx = False
    if task_ctx:
        high_count = sum(1 for t in task_ctx if t.get("priority") == "High")
        st.markdown(
            f'<div class="context-banner">🔗 <strong>Context from Task Prioritizer Agent</strong>'
            f'<br>{len(task_ctx)} tasks imported — {high_count} marked High priority.</div>',
            unsafe_allow_html=True,
        )
        use_task_ctx = st.checkbox(
            "✅ Import prioritised task list into Day Planner",
            value=True,
            help="The planner will honour the priority ranking from the Task Prioritizer Agent.",
        )

    col1, col2 = st.columns(2)
    with col1:
        start_time = st.time_input(
            "Work day starts", value=None,
            help="Your scheduled start time.",
        )
        end_time = st.time_input(
            "Work day ends", value=None,
            help="Your scheduled end time.",
        )
    with col2:
        break_mins: int = st.number_input(
            "Lunch / break (minutes)", 0, 120, 30, 15,
            help="Total break time; the agent places it at an optimal point.",
        )
        energy: str = st.selectbox(
            "Energy pattern",
            [
                "Morning person (peak focus AM)",
                "Afternoon person (peak focus PM)",
                "Consistent throughout",
            ],
            help="Determines when Deep Work blocks are scheduled.",
        )

    tasks_for_day: str = st.text_area(
        "Additional tasks for today (one per line)",
        placeholder="Write project proposal — 90 min\nTeam standup — 15 min",
        height=140,
        help="Include estimated durations where known (e.g. 'Review PRs — 1 hour').",
    )

    if st.button("▶ Run Day Planner Agent", type="primary"):
        if not start_time or not end_time:
            st.warning("⚠️ Please set both start and end times.")
        elif not tasks_for_day.strip() and not (use_task_ctx and task_ctx):
            st.warning("⚠️ Please enter tasks or import from the Task Prioritizer Agent.")
        else:
            with st.spinner("📅 Day Planner Agent is building your schedule…"):
                imported = ""
                if use_task_ctx and task_ctx:
                    imported = "\nIMPORTED FROM TASK PRIORITIZER AGENT:\n" + "\n".join(
                        f"- [{t.get('priority','?')} / {t.get('quadrant','?')}] {t.get('task','')}"
                        for t in task_ctx
                    )

                prompt = (
                    f"Build a time-blocked daily schedule:\n"
                    f"- Work hours: {start_time.strftime('%H:%M')} to {end_time.strftime('%H:%M')}\n"
                    f"- Break: {break_mins} minutes\n"
                    f"- Energy pattern: {energy}\n"
                    f"{imported}\n"
                    f"Additional tasks:\n{sanitize_input(tasks_for_day) or 'None'}\n\n"
                    "Rules:\n"
                    "- Deep Work during peak energy\n"
                    "- Admin/shallow tasks in low-energy windows\n"
                    "- Honour priority ranking if provided\n"
                    "- List overflow if schedule is full\n\n"
                    'Respond ONLY with valid JSON:\n'
                    '{"schedule":[{"time":"09:00-10:30","task":"...","type":"Deep Work"}],'
                    '"overflow_tasks":[],"tip":"productivity tip"}\n'
                    'Valid types: "Deep Work","Meeting","Admin","Break"'
                )
                try:
                    data = parse_json(planner_agent.call(prompt, AVAILABLE_MODELS))
                    st.session_state.agent_memory["last_schedule"] = data
                    log_agent_action(
                        "DAY-PLANNER AGENT",
                        f"Built schedule with {len(data.get('schedule', []))} blocks",
                    )

                    st.subheader(f"🗓️ Schedule — {date.today().strftime('%d %b %Y')}")
                    type_colors = {
                        "Deep Work": "#7c3aed", "Meeting": "#0ea5e9",
                        "Admin":     "#f59e0b", "Break":   "#10b981",
                    }
                    for block in data.get("schedule", []):
                        btype = block.get("type", "Admin")
                        color = type_colors.get(btype, "#888")
                        st.markdown(
                            f'<div class="result-card" style="border-left:4px solid {color}">'
                            f'<span style="color:#aaa;font-size:.85rem">🕐 {block.get("time","")}</span>'
                            f'&nbsp;<span class="badge" style="background:{color}22;color:{color};'
                            f'border:1px solid {color}55">{btype}</span>'
                            f'<br><strong style="font-size:.95rem">{block.get("task","")}</strong>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    overflow = data.get("overflow_tasks", [])
                    if overflow:
                        st.markdown("#### ⏭️ Overflow — move to tomorrow")
                        for t in overflow:
                            st.markdown(f"- {t}")

                    if tip := data.get("tip", ""):
                        st.info(f"💡 **Agent Tip:** {tip}")

                except Exception as exc:
                    st.error(f"Agent failed: {exc}")