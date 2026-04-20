# WorkMind AI — Intelligent Productivity Assistant

## Chosen Vertical
**Productivity / Work**

WorkMind AI is a multi-agent productivity suite that helps professionals manage tasks, meetings, emails, and daily schedules — all powered by the Google Gemini API.

---

## Approach and Logic

The application is built on a **multi-agent architecture** where each feature is an independent AI agent with its own specialised persona and system prompt.

```
WorkMind AI
├── 🗂️  TASK-PRIORITIZER AGENT   — Eisenhower Matrix specialist
├── 📝  MEETING-SUMMARIZER AGENT  — Meeting intelligence specialist
├── ✉️  EMAIL-DRAFTER AGENT       — Business communication specialist
└── 📅  DAY-PLANNER AGENT         — Schedule optimisation specialist
```

**Key design decisions:**
- Each agent is a Python class instance with a distinct `system_prompt` that is prepended to every Gemini API call, giving each agent a unique and consistent persona.
- All agents share a **common session memory** (`st.session_state`) so they can pass context to one another — for example, the Meeting Summarizer Agent's output is automatically offered as context to the Email Drafter Agent.
- Every agent instructs Gemini to return **strict JSON output**. The app parses this JSON and renders it as a structured, interactive UI — not raw text.
- A **cross-agent activity log** is maintained in the sidebar to show judges the inter-agent communication happening in real time.

**Prompt Engineering Techniques Used:**
- **Persona Prompting:** Each agent has a unique role identity injected at the system level.
- **Structured Output Enforcement:** Every prompt ends with a strict JSON schema instruction.
- **Cross-Agent Context Injection:** The Email Drafter and Day Planner agents receive structured output from other agents as additional context in their prompts.
- **Eisenhower Matrix Reasoning:** The Task Prioritizer agent applies a well-known decision framework purely through prompt logic.
- **Energy-Aware Scheduling:** The Day Planner uses cognitive load theory through prompt constraints to schedule deep work during peak energy windows.

---

## How the Solution Works

**Technology Stack:**
- **Frontend:** Streamlit (Python)
- **AI Backend:** Google Gemini API via `google-generativeai` SDK
- **Deployment:** Google Cloud Run (serverless, containerised)

**Gemini API Integration:**
1. On startup, the app calls `genai.list_models()` to dynamically discover which models are available for the provided API key. This makes the app resilient to model deprecations.
2. Each agent call prepends a specialised system prompt to the user's query before sending it to Gemini.
3. Gemini's response is parsed from JSON and rendered as a rich Streamlit UI (cards, metrics, tables).
4. A quota-aware retry loop (waits 20s → 40s → 65s) handles free-tier rate limits gracefully with a live countdown visible to the user.

**Cross-Agent Data Flow:**
```
User runs Meeting Summarizer Agent
        ↓
Summary, decisions, action items saved to session_state
        ↓
User switches to Email Drafter Agent
        ↓
Context banner appears: "Context available from Meeting Summarizer"
        ↓
Agent injects meeting context into the email prompt
        ↓
Gemini writes a highly specific, context-aware email
```

**Security:**
- The `GEMINI_API_KEY` is loaded exclusively from environment variables (Cloud Run) or `st.secrets` (local). It is never hardcoded in the source.

---

## Any Assumptions Made

- **API Key:** A valid Google AI Studio `GEMINI_API_KEY` must be provided as an environment variable. The free tier (15 RPM) is sufficient for demo purposes.
- **Model Availability:** The app uses `genai.list_models()` at startup to auto-detect available models, so it does not depend on any specific model name being available.
- **Cloud Run Deployment:** The app is containerised via Docker and designed for stateless, serverless execution on Google Cloud Run. Each container instance is independent; session state is per-user session.
- **Single User Demo:** The session state memory is per-browser-session and is not persisted to a database. In a production environment, this would be backed by Firestore or another persistent store.
- **Meeting Notes Quality:** The Meeting Summarizer Agent performs best with notes of at least 3-4 sentences containing identifiable decisions or action items.