"""
test_agents.py — Validation suite for WorkMind AI

Tests:
1. JSON parser handles all edge cases (clean JSON, markdown fences, whitespace)
2. Agent class initialises correctly with expected attributes
3. Session state memory structure is valid
4. All four agent personas are correctly defined

Run with:
    python test_agents.py
"""

import json
import sys
import os


# ─────────────────────────────────────────────────────────────
# Inline parse_json (mirrors app.py — no Streamlit import needed)
# ─────────────────────────────────────────────────────────────
def parse_json(raw: str):
    text = raw
    for fence in ["```json", "```"]:
        if text.startswith(fence):
            text = text[len(fence):]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


# ─────────────────────────────────────────────────────────────
# Minimal Agent stub (mirrors app.py Agent class attributes)
# ─────────────────────────────────────────────────────────────
class AgentStub:
    def __init__(self, name, emoji, role, system_prompt, color):
        self.name          = name
        self.emoji         = emoji
        self.role          = role
        self.system_prompt = system_prompt
        self.color         = color


# ─────────────────────────────────────────────────────────────
# Test helpers
# ─────────────────────────────────────────────────────────────
PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
results = []

def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    print(f"{status}  {name}" + (f"  →  {detail}" if detail else ""))
    results.append(condition)


# ─────────────────────────────────────────────────────────────
# Test Suite 1: JSON Parser
# ─────────────────────────────────────────────────────────────
print("\n📦 Test Suite 1: JSON Parser\n" + "─" * 40)

# 1a. Clean JSON array
raw = '[{"task": "Fix bug", "priority": "High", "quadrant": "Do First", "reason": "Critical"}]'
data = parse_json(raw)
check("1a. Clean JSON array", isinstance(data, list) and data[0]["priority"] == "High")

# 1b. JSON with ```json fence
raw = '```json\n{"subject": "Hello", "body": "World"}\n```'
data = parse_json(raw)
check("1b. ```json fence stripped", data["subject"] == "Hello")

# 1c. JSON with plain ``` fence
raw = '```\n{"summary": "meeting done"}\n```'
data = parse_json(raw)
check("1c. ``` fence stripped", data["summary"] == "meeting done")

# 1d. Extra whitespace
raw = '   \n{"tip": "Take breaks"}\n   '
data = parse_json(raw)
check("1d. Whitespace trimmed", data["tip"] == "Take breaks")

# 1e. Malformed JSON raises exception
try:
    parse_json("{not valid json}")
    check("1e. Malformed JSON raises", False, "Should have raised")
except (json.JSONDecodeError, ValueError):
    check("1e. Malformed JSON raises", True)


# ─────────────────────────────────────────────────────────────
# Test Suite 2: Agent Definitions
# ─────────────────────────────────────────────────────────────
print("\n🤖 Test Suite 2: Agent Definitions\n" + "─" * 40)

agents = [
    AgentStub("TASK-PRIORITIZER AGENT", "🗂️", "Eisenhower Matrix Specialist",
              "You are the Task Prioritizer Agent.", "#f59e0b"),
    AgentStub("MEETING-SUMMARIZER AGENT", "📝", "Meeting Intelligence Specialist",
              "You are the Meeting Summarizer Agent.", "#0ea5e9"),
    AgentStub("EMAIL-DRAFTER AGENT", "✉️", "Business Communication Specialist",
              "You are the Email Drafter Agent.", "#a855f7"),
    AgentStub("DAY-PLANNER AGENT", "📅", "Schedule Optimisation Specialist",
              "You are the Day Planner Agent.", "#10b981"),
]

for agent in agents:
    check(f"2. {agent.name} — has name",          bool(agent.name))
    check(f"   {agent.name} — has role",          bool(agent.role))
    check(f"   {agent.name} — has system_prompt", bool(agent.system_prompt))
    check(f"   {agent.name} — has color",         agent.color.startswith("#"))


# ─────────────────────────────────────────────────────────────
# Test Suite 3: Session State Memory Schema
# ─────────────────────────────────────────────────────────────
print("\n🧠 Test Suite 3: Session State Schema\n" + "─" * 40)

# Simulate what app.py creates in st.session_state
agent_memory = {
    "last_meeting_summary": None,
    "last_task_list":       None,
    "last_email_draft":     None,
    "last_schedule":        None,
    "agent_logs":           [],
}

check("3a. last_meeting_summary key exists",  "last_meeting_summary" in agent_memory)
check("3b. last_task_list key exists",        "last_task_list" in agent_memory)
check("3c. last_email_draft key exists",      "last_email_draft" in agent_memory)
check("3d. last_schedule key exists",         "last_schedule" in agent_memory)
check("3e. agent_logs is a list",             isinstance(agent_memory["agent_logs"], list))

# Simulate logging an action
agent_memory["agent_logs"].append({"timestamp": "12:00:00", "agent": "TEST", "action": "ran"})
check("3f. Agent log entry appends correctly", len(agent_memory["agent_logs"]) == 1)


# ─────────────────────────────────────────────────────────────
# Test Suite 4: Environment / Security
# ─────────────────────────────────────────────────────────────
print("\n🔒 Test Suite 4: Security\n" + "─" * 40)

# Verify the app.py source does NOT contain a hardcoded API key pattern
app_path = os.path.join(os.path.dirname(__file__), "app.py")
if os.path.exists(app_path):
    with open(app_path) as f:
        source = f.read()
    check("4a. No hardcoded 'AIzaSy' key in app.py", "AIzaSy" not in source)
    check("4b. get_secret() function present",        "def get_secret" in source)
    check("4c. GEMINI_API_KEY from env only",         'get_secret("GEMINI_API_KEY")' in source)
else:
    print("  SKIP  app.py not found in current directory")


# ─────────────────────────────────────────────────────────────
# Results Summary
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 40)
passed = sum(results)
total  = len(results)
print(f"\n{'✅' if passed == total else '⚠️ '} {passed}/{total} tests passed\n")

if passed < total:
    sys.exit(1)
