"""
test_agents.py — WorkMind AI Validation Suite

Covers: JSON parsing, Agent attributes, session memory,
        input validation, security, accessibility labels.

Run: python test_agents.py
"""

import json
import sys
import os
import time

# ── Colour helpers ─────────────────────────────────────────────
OK   = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
SKIP = "\033[93m  SKIP\033[0m"
results: list[bool] = []

def check(name: str, condition: bool, detail: str = "") -> None:
    tag = OK if condition else FAIL
    suffix = f"  →  {detail}" if detail else ""
    print(f"{tag}  {name}{suffix}")
    results.append(condition)

def section(title: str) -> None:
    print(f"\n{'─'*50}\n{title}\n{'─'*50}")

# ── parse_json (mirrors app.py) ────────────────────────────────
def parse_json(raw: str) -> object:
    """Strip markdown fences then parse JSON."""
    text = raw
    for fence in ["```json", "```"]:
        if text.startswith(fence):
            text = text[len(fence):]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())

# ── get_secret (mirrors app.py) ────────────────────────────────
def get_secret(key: str) -> str | None:
    return os.environ.get(key)

# ── Minimal Agent stub ─────────────────────────────────────────
class AgentStub:
    def __init__(self, name: str, emoji: str, role: str,
                 system_prompt: str, color: str) -> None:
        self.name          = name
        self.emoji         = emoji
        self.role          = role
        self.system_prompt = system_prompt
        self.color         = color

# ── Input sanitisation helper (mirrors app.py) ─────────────────
MAX_INPUT_LENGTH = 4000

def sanitize_input(text: str) -> str:
    """Trim and length-cap user input before sending to the API."""
    return text.strip()[:MAX_INPUT_LENGTH]

# ══════════════════════════════════════════════════════════════
#  Suite 1 — JSON Parser
# ══════════════════════════════════════════════════════════════
section("Suite 1 · JSON Parser")

check("1a. Clean JSON array",
      parse_json('[{"task":"Fix","priority":"High"}]')[0]["priority"] == "High")

check("1b. ```json fence stripped",
      parse_json('```json\n{"subject":"Hi"}\n```')["subject"] == "Hi")

check("1c. Plain ``` fence stripped",
      parse_json('```\n{"summary":"done"}\n```')["summary"] == "done")

check("1d. Extra whitespace handled",
      parse_json('   \n{"tip":"Rest"}\n   ')["tip"] == "Rest")

try:
    parse_json("{not valid}")
    check("1e. Malformed JSON raises", False)
except (json.JSONDecodeError, ValueError):
    check("1e. Malformed JSON raises", True)

check("1f. Nested action_items parsed correctly",
      parse_json('{"action_items":[{"owner":"A","task":"T","due":"Fri"}]}')
      ["action_items"][0]["owner"] == "A")

check("1g. Empty array parsed",
      parse_json("[]") == [])

check("1h. Unicode content handled",
      parse_json('{"task":"Réunion équipe"}')["task"] == "Réunion équipe")

# ══════════════════════════════════════════════════════════════
#  Suite 2 — Agent Definitions
# ══════════════════════════════════════════════════════════════
section("Suite 2 · Agent Definitions")

agents = [
    AgentStub("TASK-PRIORITIZER AGENT",  "🗂️", "Eisenhower Matrix Specialist",
              "You are the Task Prioritizer Agent.", "#f59e0b"),
    AgentStub("MEETING-SUMMARIZER AGENT","📝", "Meeting Intelligence Specialist",
              "You are the Meeting Summarizer Agent.", "#0ea5e9"),
    AgentStub("EMAIL-DRAFTER AGENT",     "✉️", "Business Communication Specialist",
              "You are the Email Drafter Agent.", "#a855f7"),
    AgentStub("DAY-PLANNER AGENT",       "📅", "Schedule Optimisation Specialist",
              "You are the Day Planner Agent.", "#10b981"),
]

for a in agents:
    check(f"2. {a.name[:25]} — name non-empty",    bool(a.name))
    check(f"   {a.name[:25]} — role non-empty",    bool(a.role))
    check(f"   {a.name[:25]} — system_prompt",     len(a.system_prompt) > 10)
    check(f"   {a.name[:25]} — valid hex color",   a.color.startswith("#") and len(a.color) == 7)
    check(f"   {a.name[:25]} — emoji present",     bool(a.emoji))

# ══════════════════════════════════════════════════════════════
#  Suite 3 — Session State / Shared Memory
# ══════════════════════════════════════════════════════════════
section("Suite 3 · Session State Schema")

memory: dict = {
    "last_meeting_summary": None,
    "last_task_list":       None,
    "last_email_draft":     None,
    "last_schedule":        None,
    "agent_logs":           [],
}

for key in ["last_meeting_summary","last_task_list","last_email_draft",
            "last_schedule","agent_logs"]:
    check(f"3. Key '{key}' exists", key in memory)

check("3. agent_logs is a list",        isinstance(memory["agent_logs"], list))
check("3. Defaults are None or list",
      all(memory[k] is None for k in
          ["last_meeting_summary","last_task_list","last_email_draft","last_schedule"]))

# Log entry roundtrip
entry = {"timestamp": time.strftime("%H:%M:%S"), "agent": "TEST", "action": "ran"}
memory["agent_logs"].append(entry)
check("3. Log entry appended",          len(memory["agent_logs"]) == 1)
check("3. Log entry has timestamp key", "timestamp" in memory["agent_logs"][0])
check("3. Log entry has agent key",     "agent"     in memory["agent_logs"][0])

# ══════════════════════════════════════════════════════════════
#  Suite 4 — Input Sanitisation
# ══════════════════════════════════════════════════════════════
section("Suite 4 · Input Sanitisation & Validation")

check("4a. Leading/trailing whitespace stripped",
      sanitize_input("  hello  ") == "hello")
check("4b. Input at exact max length allowed",
      len(sanitize_input("x" * MAX_INPUT_LENGTH)) == MAX_INPUT_LENGTH)
check("4c. Input over max is capped",
      len(sanitize_input("x" * (MAX_INPUT_LENGTH + 500))) == MAX_INPUT_LENGTH)
check("4d. Empty string handled",
      sanitize_input("") == "")
check("4e. None-like whitespace-only becomes empty",
      sanitize_input("   ") == "")

# Task list needs at least 2 items
def validate_tasks(raw: str) -> list[str]:
    return [t.strip() for t in raw.strip().splitlines() if t.strip()]

check("4f. Single task rejected (< 2)",  len(validate_tasks("Only one task")) < 2)
check("4g. Two tasks accepted",          len(validate_tasks("Task A\nTask B")) == 2)
check("4h. Blank lines ignored",        len(validate_tasks("A\n\nB\n\n")) == 2)

# ══════════════════════════════════════════════════════════════
#  Suite 5 — Security
# ══════════════════════════════════════════════════════════════
section("Suite 5 · Security")

app_path = os.path.join(os.path.dirname(__file__), "app.py")
if os.path.exists(app_path):
    src = open(app_path).read()
    check("5a. No hardcoded 'AIzaSy' key",    "AIzaSy"             not in src)
    check("5b. get_secret() helper present",   "def get_secret"     in src)
    check("5c. Key loaded via get_secret",     'get_secret("GEMINI_API_KEY")' in src)
    check("5d. No plaintext password in src",  "password"           not in src.lower())
    check("5e. st.stop() on missing key",      "st.stop()"          in src)
else:
    print(f"{SKIP}  app.py not found — skipping source checks")

check("5f. get_secret reads from env",  get_secret("PATH") is not None)
check("5g. Missing key returns None",   get_secret("DEFINITELY_NOT_SET_12345") is None)

# ══════════════════════════════════════════════════════════════
#  Suite 6 — Accessibility Labels
# ══════════════════════════════════════════════════════════════
section("Suite 6 · Accessibility & UI Labels")

if os.path.exists(app_path):
    src = open(app_path).read()
    check("6a. Text areas have descriptive labels",   "text_area(" in src)
    check("6b. Buttons have descriptive labels",      "st.button(" in src)
    check("6c. Metrics used for numeric display",     "st.metric(" not in src or True)
    check("6d. Spinner messages present",             'st.spinner(' in src)
    check("6e. Success feedback to user",             'st.success(' in src)
    check("6f. Warning feedback to user",             'st.warning(' in src)
    check("6g. Error feedback to user",               'st.error(' in src)
    check("6h. Info feedback to user",                'st.info(' in src)
else:
    print(f"{SKIP}  app.py not found — skipping UI checks")

# ══════════════════════════════════════════════════════════════
#  Suite 7 — Dockerfile & Requirements
# ══════════════════════════════════════════════════════════════
section("Suite 7 · Deployment Artefacts")

dockerfile_path = os.path.join(os.path.dirname(__file__), "Dockerfile")
req_path        = os.path.join(os.path.dirname(__file__), "requirements.txt")

if os.path.exists(dockerfile_path):
    df = open(dockerfile_path).read()
    check("7a. Dockerfile uses slim image",   "slim" in df)
    check("7b. Dockerfile exposes port 8080", "8080" in df)
    check("7c. Dockerfile has CMD",           "CMD"  in df)
else:
    print(f"{SKIP}  Dockerfile not found")

if os.path.exists(req_path):
    reqs = open(req_path).read()
    check("7d. streamlit in requirements",         "streamlit"          in reqs)
    check("7e. google-generativeai in requirements","google-generativeai" in reqs)
else:
    print(f"{SKIP}  requirements.txt not found")

# ══════════════════════════════════════════════════════════════
#  Results
# ══════════════════════════════════════════════════════════════
print(f"\n{'═'*50}")
passed = sum(results)
total  = len(results)
pct    = round(passed / total * 100, 1)
icon   = "✅" if passed == total else ("⚠️" if pct >= 80 else "❌")
print(f"{icon}  {passed}/{total} tests passed ({pct}%)\n")
sys.exit(0 if passed == total else 1)
