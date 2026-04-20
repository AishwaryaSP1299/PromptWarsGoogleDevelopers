<div align="center">
  <img src="https://raw.githubusercontent.com/google/material-design-icons/master/symbols/web/smart_toy/materialsymbolsoutlined/smart_toy_wght400grad0opsz48.png" width="80" alt="WorkMind AI Logo">
  <h1>🧠 WorkMind AI</h1>
  <p><em>Enterprise-Grade Multi-Agent Productivity Suite powered by Google Gemini</em></p>
  
  [![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
  [![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
  [![Google Gemini API](https://img.shields.io/badge/Google_Gemini-API-8E75B2?logo=google&logoColor=white)](https://aistudio.google.com/)
  [![PyTest Coverage](https://img.shields.io/badge/PyTest-70%2B_Assertions-success?logo=pytest&logoColor=white)](https://pytest.org)
</div>

---

## 🌟 The Vision: Beyond Chatbots

WorkMind AI is not a chatbot — it is a **concurrent, multi-agent orchestrator** designed for the modern enterprise. We have moved past simple Q&A to build a system where specialized AI personas collaborate, share memory context, and execute complex workflows while strictly adhering to corporate data privacy constraints.

**Chosen Vertical:** Productivity / Work

---

## 🏛️ Architecture: The 4-Agent Ecosystem

The application is built on a robust `AgentOrchestrator` that fans out parallel API calls to four distinct, hyper-specialized agents:

```mermaid
graph TD
    User([User Input]) --> Orchestrator{Agent Orchestrator\n(Concurrent Dispatch)}
    
    Orchestrator --> A[🗂️ Task Prioritizer Agent]
    Orchestrator --> B[📝 Meeting Summarizer Agent]
    Orchestrator --> C[✉️ Email Drafter Agent]
    Orchestrator --> D[📅 Day Planner Agent]
    
    A -.->|Injects Context| D
    B -.->|Injects Context| C
    
    A --> UI[Streamlit UI]
    B --> UI
    C --> UI
    D --> UI
    
    subgraph Enterprise Guardrails
        DP[DataPrivacy Engine\nPII Redaction & Injection Defense]
    end
    
    User -.-> DP
    DP -.-> Orchestrator
```

### 🧠 The Agents
1. **🗂️ Task Prioritizer Agent:** Applies the Eisenhower Matrix to strictly rank tasks by urgency × impact.
2. **📝 Meeting Summarizer Agent:** Parses chaotic raw notes into executive summaries, decisions, and action items.
3. **✉️ Email Drafter Agent:** Crafts tone-aware, intent-driven business communications.
4. **📅 Day Planner Agent:** Uses Cognitive Load Theory to map Deep Work blocks to your biological peak energy windows.

---

## 🚀 How We Maxed Out The Evaluation Criteria

This codebase isn't just a prototype; it's engineered to pass rigorous enterprise AI/MLOps standards. Here is exactly how we tackled the 6 evaluation pillars:

### 1. 🛡️ Security (Data Privacy & Prompt Guardrails)
- **PII Redaction Engine:** The custom `DataPrivacy` class intercepts all user inputs and uses regex to scrub SSNs, Credit Cards, Emails, and Phone Numbers (replacing them with `[REDACTED]` tokens) *before* they ever reach the Gemini API.
- **Prompt Injection Defense:** Scans for known jailbreak vectors (e.g., *"ignore previous instructions"*).
- **Secrets Management:** `GEMINI_API_KEY` is strictly loaded via OS env vars or Streamlit secrets. The `.gitignore` prevents credential leaks.

### 2. ⚡ Efficiency (Asynchronous Orchestration)
- **Concurrent Dispatch:** Instead of blocking sequentially, the `AgentOrchestrator` uses Python's `ThreadPoolExecutor` to fan out multi-agent requests concurrently, drastically reducing I/O-bound latency.
- **Dynamic Model Discovery:** Uses `@st.cache_resource` on `genai.list_models()` to dynamically detect available Gemini models, caching the result to eliminate redundant API handshakes.

### 3. 🧪 Testing (Production-Grade PyTest Suite)
- **70+ Assertions across 10 Suites:** A massive `test_agents.py` suite covering JSON sanitization, agent properties, UI accessibility, security statics, memory schemas, PII redaction, and concurrent orchestration.
- **Mocking:** Uses `unittest.mock.patch` to isolate and test `Agent.call()` logic without hitting the live Gemini API.
- **Custom Exceptions:** A fully typed custom exception hierarchy (`exceptions.py`) replaces bare exceptions for precise error tracking.

### 4. 💎 Code Quality
- **Strict Typing:** 100% Python type hints (`-> str | None`, `tuple[Agent, ...]`).
- **Modularity:** Separated into `app.py` (UI), `agents.py` (Business Logic), `utils.py` (Security & Logs), and `exceptions.py` (Error States).
- **Tooling:** Controlled via `pyproject.toml` with strict `ruff` linting configurations.

### 5. ♿ Accessibility & Usability
- **Semantic HTML & ARIA:** Agents render their headers with `role="banner"` and `aria-label` tags.
- **Universal Feedback:** Every interactive element has `help=` tooltips. Every action provides clear visual feedback via `st.spinner()`, `st.success()`, or `st.error()`.
- **Cross-Agent Memory:** The user doesn't have to copy-paste. The Meeting Summarizer's output natively populates as context for the Email Drafter.

### 6. ☁️ Google Services Integration
1. **Google Gemini API:** Advanced prompt engineering enforcing strict JSON schemas. Robust quota-aware retry loop (20s → 40s → 65s backoff) built specifically to handle Free Tier rate limits.
2. **Google Cloud Logging:** The `setup_logger()` function emits structured JSON payloads designed specifically to be automatically parsed and indexed by the Google Cloud Run logging agent, providing instant telemetry without an extra SDK.

---

## 🛠️ Installation & Execution

### Local Development
```bash
# 1. Clone and install
git clone https://github.com/AishwaryaSP1299/PromptWarsGoogleDevelopers.git
cd PromptWarsGoogleDevelopers
pip install -r requirements.txt

# 2. Set your Gemini API Key
mkdir -p .streamlit
echo 'GEMINI_API_KEY = "your_api_key_here"' > .streamlit/secrets.toml

# 3. Run the application
streamlit run app.py
```

### Run the Test Suite
```bash
pytest test_agents.py -v
```

### Google Cloud Run Deployment
```bash
gcloud run deploy workmind-ai \
  --source . \
  --project YOUR_PROJECT_ID \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="GEMINI_API_KEY=your_actual_key"
```

---

## 🧠 Prompt Engineering Deep Dive

We bypassed standard conversational prompting in favor of strict, JSON-first architectural constraints:
- **System-Level Persona Injection:** Each agent has an immutable core identity.
- **Forced Schema Parsing:** Prompts demand array/object JSON structures (`{"subject":"...","body":"..."}`). The `parse_json()` utility strips markdown fences, ensuring the UI always receives pure data objects instead of raw conversational text.

## ⚠️ Assumptions Made
- **Cloud Run Execution:** The app assumes it is deployed on a stateless container. Session memory is tied to the active Streamlit browser session.
- **Model Fallback:** The dynamic discovery logic assumes at least one `gemini-pro` or `gemini-flash` model is available to the configured API key.

---
*Built for the Google Developers PromptWars Hackathon.*