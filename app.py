import streamlit as st
import google.generativeai as genai
import time
import os
import json
from datetime import date

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WorkMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .agent-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #181825 100%);
        border: 1px solid #313244;
        border-radius: 16px;
        padding: 1rem 1.4rem;
        margin-bottom: 1rem;
    }
    .agent-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 0.3rem;
    }
    .agent-name {
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #89b4fa;
    }
    .agent-status {
        font-size: 0.72rem;
        padding: 2px 8px;
        border-radius: 20px;
        background: #21c45422;
        color: #21c454;
        border: 1px solid #21c45455;
    }
    .result-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
    }
    .context-banner {
        background: #7c3aed22;
        border: 1px solid #7c3aed55;
        border-radius: 10px;
        padding: 0.7rem 1rem;
        margin-bottom: 1rem;
        font-size: 0.85rem;
        color: #c4b5fd;
    }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-high   { background:#ff4b4b22; color:#ff4b4b; border:1px solid #ff4b4b55; }
    .badge-medium { background:#ffa50022; color:#ffa500; border:1px solid #ffa50055; }
    .badge-low    { background:#21c45422; color:#21c454; border:1px solid #21c45455; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Session state — shared memory across agents
# ─────────────────────────────────────────────────────────────
if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = {
        "last_meeting_summary": None,    # set by Meeting Summarizer Agent
        "last_task_list":       None,    # set by Task Prioritizer Agent
        "last_email_draft":     None,    # set by Email Drafter Agent
        "last_schedule":        None,    # set by Day Planner Agent
        "agent_logs":           [],      # cross-agent activity log
    }

def log_agent_action(agent_name: str, action: str):
    """Append an entry to the shared cross-agent log."""
    st.session_state.agent_memory["agent_logs"].append({
        "timestamp": time.strftime("%H:%M:%S"),
        "agent": agent_name,
        "action": action,
    })

# ─────────────────────────────────────────────────────────────
# Secret helper
# ─────────────────────────────────────────────────────────────
def get_secret(key: str):
    if key in os.environ:
        return os.environ[key]
    has_file = any(os.path.exists(p) for p in [
        ".streamlit/secrets.toml",
        "/root/.streamlit/secrets.toml",
        "/app/.streamlit/secrets.toml",
    ])
    if has_file and hasattr(st, "secrets"):
        try:
            return st.secrets.get(key)
        except Exception:
            pass
    return None

# ─────────────────────────────────────────────────────────────
# Gemini setup
# ─────────────────────────────────────────────────────────────
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("🔑 **GEMINI_API_KEY not found.** Pass it via --set-env-vars during Cloud Run deploy.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

@st.cache_resource(show_spinner="🔍 Detecting available Gemini models…")
def get_available_models():
    try:
        available = [
            m.name for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]
        preferred = ["flash", "pro"]
        def sort_key(name):
            for i, p in enumerate(preferred):
                if p in name:
                    return i
            return 99
        available.sort(key=sort_key)
        return available
    except Exception as e:
        st.error(f"Could not list models: {e}")
        return []

AVAILABLE_MODELS = get_available_models()

if not AVAILABLE_MODELS:
    st.error("No usable Gemini models found for your API key.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# Base Agent Class
# ─────────────────────────────────────────────────────────────
class Agent:
    """
    A single AI agent with its own identity, system prompt, and memory.
    All agents share the global session state for cross-agent context.
    """
    def __init__(self, name: str, emoji: str, role: str, system_prompt: str, color: str):
        self.name         = name
        self.emoji        = emoji
        self.role         = role
        self.system_prompt = system_prompt
        self.color        = color

    def call(self, user_prompt: str, max_retries: int = 3) -> str:
        """
        Send a prompt to this agent (prepends its system persona).
        Falls back through available models on 404. Retries on quota errors.
        """
        full_prompt = f"{self.system_prompt}\n\n{user_prompt}"
        wait_schedule = [20, 40, 65]

        for attempt in range(max_retries + 1):
            last_404_error = None
            for m_name in AVAILABLE_MODELS:
                try:
                    model = genai.GenerativeModel(m_name)
                    response = model.generate_content(full_prompt)
                    log_agent_action(self.name, f"Generated response using {m_name}")
                    return response.text.strip()
                except Exception as e:
                    err = str(e)
                    if "404" in err or "not found" in err.lower():
                        last_404_error = err
                        continue
                    if any(c in err for c in ["429", "TooManyRequests", "503", "quota"]):
                        break
                    raise
            else:
                if last_404_error:
                    raise Exception(f"No usable Gemini model found. Last error: {last_404_error}")

            if attempt < max_retries:
                wait = wait_schedule[attempt]
                ph = st.empty()
                for remaining in range(wait, 0, -1):
                    ph.warning(
                        f"⏳ **{self.name}** is rate-limited. Retrying in {remaining}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(1)
                ph.empty()
            else:
                raise Exception(f"**{self.name}**: API Quota exceeded. Please wait a minute and try again.")

    def render_header(self, status: str = "ONLINE"):
        """Render a styled agent header card in the UI."""
        st.markdown(
            f"""<div class="agent-header">
                <span style="font-size:1.4rem">{self.emoji}</span>
                <span class="agent-name">{self.name}</span>
                <span class="agent-status">● {status}</span>
                <span style="font-size:0.78rem; color:#6c7086; margin-left:4px">{self.role}</span>
            </div>""",
            unsafe_allow_html=True,
        )


def parse_json(raw: str):
    """Strip markdown fences and parse JSON."""
    text = raw
    for fence in ["```json", "```"]:
        if text.startswith(fence):
            text = text[len(fence):]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


# ─────────────────────────────────────────────────────────────
# Instantiate the four agents with distinct personas
# ─────────────────────────────────────────────────────────────
task_agent = Agent(
    name="TASK-PRIORITIZER AGENT",
    emoji="🗂️",
    role="Eisenhower Matrix Specialist",
    color="#f59e0b",
    system_prompt="""You are the Task Prioritizer Agent — an expert in productivity frameworks.
Your speciality is the Eisenhower Matrix (Do First / Schedule / Delegate / Drop).
You analyse tasks through the lens of urgency and impact.
You return ONLY valid JSON arrays. Never include preamble, markdown, or explanation.""",
)

meeting_agent = Agent(
    name="MEETING-SUMMARIZER AGENT",
    emoji="📝",
    role="Meeting Intelligence Specialist",
    color="#0ea5e9",
    system_prompt="""You are the Meeting Summarizer Agent — an expert facilitator and analyst.
Your job is to process raw, unstructured meeting notes and extract:
- A concise executive summary
- Key decisions made
- Action items with owners and deadlines
- Open questions that remain unresolved
You return ONLY valid JSON objects. Never include preamble, markdown, or explanation.""",
)

email_agent = Agent(
    name="EMAIL-DRAFTER AGENT",
    emoji="✉️",
    role="Business Communication Specialist",
    color="#a855f7",
    system_prompt="""You are the Email Drafter Agent — a world-class business communication expert.
You craft professional, tone-aware emails based on intent and recipient context.
When context from a meeting summary is provided, you use it to write highly specific, informed emails.
You return ONLY valid JSON objects with 'subject' and 'body' keys.
Never include preamble, markdown fences, or explanation outside the JSON.""",
)

planner_agent = Agent(
    name="DAY-PLANNER AGENT",
    emoji="📅",
    role="Schedule Optimisation Specialist",
    color="#10b981",
    system_prompt="""You are the Day Planner Agent — an expert productivity coach and scheduler.
You build optimised, time-blocked schedules aligned to human energy patterns and cognitive load theory.
You always schedule Deep Work during peak energy windows and cluster shallow tasks in low-energy slots.
You return ONLY valid JSON objects. Never include preamble, markdown, or explanation.""",
)

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 WorkMind AI")
    st.caption("Multi-Agent Productivity Suite")
    st.markdown("---")
    tool = st.radio(
        label="Agent",
        options=[
            "🗂️  Task Prioritizer",
            "📝  Meeting Summarizer",
            "✉️  Email Drafter",
            "📅  Day Planner",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # Cross-agent activity log
    st.markdown("#### 🔗 Agent Activity Log")
    logs = st.session_state.agent_memory.get("agent_logs", [])
    if logs:
        for entry in reversed(logs[-6:]):
            st.caption(f"`{entry['timestamp']}` **{entry['agent']}** — {entry['action']}")
    else:
        st.caption("No agent activity yet.")

    st.markdown("---")
    st.caption(f"📆 {date.today().strftime('%A, %d %B %Y')}")
    st.caption("Powered by Google Gemini")

# ─────────────────────────────────────────────────────────────
# AGENT 1 — TASK PRIORITIZER
# ─────────────────────────────────────────────────────────────
if tool == "🗂️  Task Prioritizer":
    task_agent.render_header()
    st.markdown(
        "Submit your task list and this agent will rank them by **urgency × impact** "
        "using the Eisenhower Matrix."
    )

    tasks_input = st.text_area(
        "Your tasks — one per line",
        placeholder="Respond to client email\nFix production bug\nPrepare weekly report\nReview pull requests",
        height=180,
    )
    context_input = st.text_input(
        "Context for the agent (optional)",
        placeholder="e.g. Sprint ends tomorrow, client is a key account",
    )

    if st.button("▶ Run Task Prioritizer Agent", type="primary", disabled=not tasks_input.strip()):
        tasks = [t.strip() for t in tasks_input.strip().splitlines() if t.strip()]
        if len(tasks) < 2:
            st.warning("Please enter at least 2 tasks.")
        else:
            with st.spinner(f"🗂️ Task Prioritizer Agent is working…"):
                prompt = f"""Analyse and prioritise the following tasks.
For each task assign:
- "priority": "High", "Medium", or "Low"
- "quadrant": one of "Do First", "Schedule", "Delegate", "Drop"
- "reason": one concise sentence explaining the rating

Context: {context_input or 'None'}

Tasks:
{chr(10).join(f"- {t}" for t in tasks)}

Respond ONLY with a valid JSON array:
[{{"task": "...", "priority": "High", "quadrant": "Do First", "reason": "..."}}]"""
                try:
                    results = parse_json(task_agent.call(prompt))
                    # Save to shared session state so other agents can use it
                    st.session_state.agent_memory["last_task_list"] = results
                    log_agent_action("TASK-PRIORITIZER AGENT", f"Prioritised {len(results)} tasks")

                    st.subheader("Prioritised Task List")
                    for item in results:
                        p = item.get("priority", "Medium")
                        cls = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}.get(p, "badge-medium")
                        st.markdown(
                            f"""<div class="result-card">
                            <span class="badge {cls}">{p}</span>
                            <span class="badge badge-medium">{item.get('quadrant','')}</span>
                            <strong style="font-size:1rem">&nbsp;{item.get('task','')}</strong>
                            <p style="color:#aaa;margin:.5rem 0 0;font-size:.9rem">
                            💡 {item.get('reason','')}</p>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                    st.success("✅ Results saved. The Day Planner Agent can now use these tasks.")
                except Exception as e:
                    st.error(f"Agent failed: {e}")

# ─────────────────────────────────────────────────────────────
# AGENT 2 — MEETING SUMMARIZER
# ─────────────────────────────────────────────────────────────
elif tool == "📝  Meeting Summarizer":
    meeting_agent.render_header()
    st.markdown(
        "Paste your raw meeting notes and this agent will extract a structured "
        "**summary, decisions, action items, and open questions**. "
        "The output is automatically shared with the Email Drafter Agent."
    )

    meeting_title = st.text_input("Meeting title (optional)", placeholder="e.g. Q2 Planning Sync")
    notes_input = st.text_area("Paste your meeting notes here", height=260)

    if st.button("▶ Run Meeting Summarizer Agent", type="primary", disabled=not notes_input.strip()):
        with st.spinner("📝 Meeting Summarizer Agent is processing…"):
            prompt = f"""Meeting: {meeting_title or 'Untitled'}
Notes:
{notes_input}

Respond ONLY with valid JSON:
{{
  "summary": "2-3 sentence executive overview",
  "decisions": ["decision 1", "decision 2"],
  "action_items": [{{"owner": "Name or TBD", "task": "...", "due": "deadline or TBD"}}],
  "open_questions": ["question 1"]
}}"""
            try:
                data = parse_json(meeting_agent.call(prompt))

                # Save to shared session state for Email Drafter Agent
                st.session_state.agent_memory["last_meeting_summary"] = {
                    "title": meeting_title or "Untitled Meeting",
                    "data":  data,
                }
                log_agent_action("MEETING-SUMMARIZER AGENT", f"Summarised '{meeting_title or 'Untitled'}' — context ready for Email Drafter")

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
                            f"""<div class="result-card">
                            <strong>{item.get('task','')}</strong><br>
                            <span style="color:#aaa;font-size:.85rem">
                            👤 {item.get('owner','TBD')} &nbsp;|&nbsp;
                            📅 {item.get('due','TBD')}</span>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                st.success("✅ Summary saved to agent memory. Switch to **Email Drafter** to use it!")
            except Exception as e:
                st.error(f"Agent failed: {e}")

# ─────────────────────────────────────────────────────────────
# AGENT 3 — EMAIL DRAFTER
# ─────────────────────────────────────────────────────────────
elif tool == "✉️  Email Drafter":
    email_agent.render_header()
    st.markdown(
        "Describe what you need and this agent drafts a **professional, context-aware email** instantly."
    )

    # Show cross-agent context if Meeting Summarizer has run
    meeting_ctx = st.session_state.agent_memory.get("last_meeting_summary")
    use_meeting_ctx = False
    if meeting_ctx:
        st.markdown(
            f"""<div class="context-banner">
            🔗 <strong>Context available from Meeting Summarizer Agent</strong><br>
            Meeting: <em>{meeting_ctx['title']}</em> — Summary: {meeting_ctx['data'].get('summary','')[:120]}…
            </div>""",
            unsafe_allow_html=True,
        )
        use_meeting_ctx = st.checkbox("✅ Use meeting summary as context for this email", value=True)

    col1, col2 = st.columns(2)
    with col1:
        recipient = st.text_input("Recipient", placeholder="e.g. Client at Acme Corp")
        tone = st.selectbox("Tone", ["Professional", "Friendly", "Assertive", "Apologetic", "Concise"])
    with col2:
        subject_hint = st.text_input("Subject hint (optional)", placeholder="e.g. Project delay update")
        length = st.selectbox("Length", ["Short (3–4 sentences)", "Medium (2 paragraphs)", "Detailed (3+ paragraphs)"])

    intent = st.text_area(
        "What do you want to communicate?",
        placeholder="e.g. Follow up on action items from the Q2 planning sync.",
        height=120,
    )

    if st.button("▶ Run Email Drafter Agent", type="primary", disabled=not intent.strip()):
        with st.spinner("✉️ Email Drafter Agent is writing…"):
            # Build cross-agent context block if available
            ctx_block = ""
            if use_meeting_ctx and meeting_ctx:
                m = meeting_ctx["data"]
                ctx_block = f"""
CONTEXT FROM MEETING SUMMARIZER AGENT:
Meeting: {meeting_ctx['title']}
Summary: {m.get('summary', '')}
Decisions: {', '.join(m.get('decisions', []))}
Action Items: {json.dumps(m.get('action_items', []))}
Open Questions: {', '.join(m.get('open_questions', []))}
"""

            prompt = f"""Draft a {tone.lower()} email with these parameters:
- Recipient: {recipient or 'the recipient'}
- Subject hint: {subject_hint or 'derive from context'}
- Length: {length}
- Intent: {intent}
{ctx_block}
Respond ONLY with valid JSON:
{{
  "subject": "Email subject line",
  "body": "Full email body including greeting and sign-off"
}}"""
            try:
                data = parse_json(email_agent.call(prompt))
                st.session_state.agent_memory["last_email_draft"] = data
                log_agent_action("EMAIL-DRAFTER AGENT", f"Drafted email: '{data.get('subject','')}'")

                st.subheader("Your Drafted Email")
                st.text_input("Subject", value=data.get("subject", ""))
                st.text_area("Body", value=data.get("body", ""), height=320)
                st.success("✅ Email drafted successfully.")
            except Exception as e:
                st.error(f"Agent failed: {e}")

# ─────────────────────────────────────────────────────────────
# AGENT 4 — DAY PLANNER
# ─────────────────────────────────────────────────────────────
elif tool == "📅  Day Planner":
    planner_agent.render_header()
    st.markdown(
        "Provide your working hours and tasks — this agent builds an "
        "**optimised, time-blocked schedule** around your energy levels."
    )

    # Show cross-agent context if Task Prioritizer has run
    task_ctx = st.session_state.agent_memory.get("last_task_list")
    use_task_ctx = False
    if task_ctx:
        high_tasks = [t["task"] for t in task_ctx if t.get("priority") == "High"]
        st.markdown(
            f"""<div class="context-banner">
            🔗 <strong>Context from Task Prioritizer Agent</strong><br>
            {len(task_ctx)} tasks available — {len(high_tasks)} marked High priority.
            </div>""",
            unsafe_allow_html=True,
        )
        use_task_ctx = st.checkbox("✅ Import prioritised task list into Day Planner", value=True)

    col1, col2 = st.columns(2)
    with col1:
        start_time = st.time_input("Work day starts", value=None)
        end_time   = st.time_input("Work day ends",   value=None)
    with col2:
        break_mins = st.number_input("Lunch/break (minutes)", 0, 120, 30, 15)
        energy = st.selectbox(
            "Energy pattern",
            ["Morning person (peak focus AM)", "Afternoon person (peak focus PM)", "Consistent throughout"],
        )

    tasks_for_day = st.text_area(
        "Additional tasks for today (one per line)",
        placeholder="Write project proposal — 90 min\nTeam standup — 15 min\nReview PRs",
        height=140,
    )

    if st.button("▶ Run Day Planner Agent", type="primary"):
        if not start_time or not end_time:
            st.warning("Please set your start and end times.")
        elif not tasks_for_day.strip() and not (use_task_ctx and task_ctx):
            st.warning("Please enter tasks or import from the Task Prioritizer Agent.")
        else:
            with st.spinner("📅 Day Planner Agent is building your schedule…"):
                # Build cross-agent context
                imported_tasks = ""
                if use_task_ctx and task_ctx:
                    imported_tasks = "\nIMPORTED FROM TASK PRIORITIZER AGENT (already ranked):\n"
                    imported_tasks += "\n".join(
                        f"- [{t.get('priority','?')} / {t.get('quadrant','?')}] {t.get('task','')}"
                        for t in task_ctx
                    )

                prompt = f"""Build a time-blocked daily schedule:
- Work hours: {start_time.strftime('%H:%M')} to {end_time.strftime('%H:%M')}
- Break/lunch: {break_mins} minutes
- Energy pattern: {energy}
{imported_tasks}
Additional tasks from user:
{tasks_for_day or 'None'}

Rules:
- Schedule Deep Work during peak energy hours
- Cluster Admin/shallow tasks in low-energy slots
- Include the break block
- Honour the priority ranking from Task Prioritizer Agent if provided
- List overflow tasks if schedule is full

Respond ONLY with valid JSON:
{{
  "schedule": [{{"time": "09:00 - 10:30", "task": "...", "type": "Deep Work"}}],
  "overflow_tasks": ["task A"],
  "tip": "One personalised productivity tip"
}}
Valid types: "Deep Work", "Meeting", "Admin", "Break"
"""
                try:
                    data = parse_json(planner_agent.call(prompt))
                    st.session_state.agent_memory["last_schedule"] = data
                    log_agent_action("DAY-PLANNER AGENT", f"Built schedule with {len(data.get('schedule', []))} blocks")

                    st.subheader(f"🗓️ Schedule — {date.today().strftime('%d %b %Y')}")

                    type_colors = {
                        "Deep Work": "#7c3aed",
                        "Meeting":   "#0ea5e9",
                        "Admin":     "#f59e0b",
                        "Break":     "#10b981",
                    }

                    for block in data.get("schedule", []):
                        btype = block.get("type", "Admin")
                        color = type_colors.get(btype, "#888")
                        st.markdown(
                            f"""<div class="result-card" style="border-left:4px solid {color}">
                            <span style="color:#aaa;font-size:.85rem">🕐 {block.get('time','')}</span>
                            &nbsp;
                            <span class="badge" style="background:{color}22;color:{color};border:1px solid {color}55">
                              {btype}
                            </span>
                            <br><strong style="font-size:.95rem">{block.get('task','')}</strong>
                            </div>""",
                            unsafe_allow_html=True,
                        )

                    overflow = data.get("overflow_tasks", [])
                    if overflow:
                        st.markdown("#### ⏭️ Overflow — move to tomorrow")
                        for t in overflow:
                            st.markdown(f"- {t}")

                    tip = data.get("tip", "")
                    if tip:
                        st.info(f"💡 **Agent Tip:** {tip}")

                except Exception as e:
                    st.error(f"Agent failed: {e}")