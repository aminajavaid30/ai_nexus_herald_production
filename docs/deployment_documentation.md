# 🚀 AI Nexus Herald: Deployment Documentation

> This deployment documentation is intended to help DevOps and platform teams deploy, configure, and maintain the **AI Nexus Herald** system with minimal effort.

---

## ✅ 1. System Purpose & Context

**AI Nexus Herald** is a multi-agent AI-powered newsletter generation system. It collects trending AI topics from RSS feeds, performs deep research on those topics by extracting news using the same RSS feeds, and generates a structured newsletter using LLMs via the **Groq API**.

- **Use Case**: Automates discovery and summarization of AI news
- **Audience**: AI/Tech professionals, researchers, newsletter subscribers
- **Mode**: Self-hosted FastAPI backend + Streamlit frontend

---

## ✅ 2. System Architecture

```plaintext
                        ┌────────────────────┐
                        │ Streamlit Frontend │
                        └─────────┬──────────┘
                                  │
                      User clicks "Generate Newsletter"
                                  │
                          ┌───────▼─────────┐
                          │ FastAPI Backend │
                          └───────┬─────────┘
                                  │
                          ┌───────▼─────────┐
                          │   Orchestrator  │
                          └───────┬─────────┘
                                  │
               ┌──────────────────┼──────────────────┐
               │                  │                  │
        ┌──────▼──────┐    ┌──────▼───────┐    ┌─────▼──────┐
        │ Topic Finder│    │ Deep Research│    │ Newsletter │
        │   Agent     │    │    Agent     │    │   Writer   │
        └─────────────┘    └──────────────┘    └────────────┘
                                  │
                           ┌──────▼─────┐
                           │  Groq LLM  │
                           └────────────┘
                                  │
                           ┌──────▼─────┐
                           │ Output .md │
                           │ Newsletter │
                           └────────────┘
```


- **LangGraph** orchestrates agent interaction  
- **Groq API** serves as the LLM backend  
- **Custom tools** handle RSS parsing and markdown saving

---

## ✅ 3. Deployment & Configuration

### 🔧 Prerequisites

- Python 3.12  
- Git, pip  
- `.env` file with API keys

### 📁 Environment Variables (`.env`)
```bash
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT="AI Nexus Herald"
```


### ⚙️ Local Setup

```bash
git clone https://github.com/aminajavaid30/ai_nexus_herald_production.git
cd ai_nexus_herald_production

# Install dependencies
pip install -r requirements.txt

# Start backend
uvicorn src.backend.main:app --reload

# Start frontend
cd src/frontend
streamlit run Home.py
```

## ✅ 4. Operations & Observability
### 📜 Logging
- Custom logger at src/backend/logger.py
- Logs saved to outputs/logs/
- Log levels: INFO, ERROR

### 💓 Health Checks
FastAPI endpoint:
```bash
@app.get("/health")
async def health():
    return {"status": "ok"}
```

### 🧯 Fallbacks
- Groq API failure handling
- RSS parsing robustness
- Newsletter generation exceptions

### 🔍 Monitoring
- LangSmith tracing for all agent and LangGraph workflows

---

## ✅ 5. Maintenance & Lifecycle
### 🔁 Regular Maintenance
- Update RSS feed list in rss_config.yaml
- Tune prompts in prompt_config.yaml
- Rotate logs from outputs/logs/
- Periodic validation using DeepEval metrics

### 🧪 Evaluation & Tests

```bash
pytest tests
```

**Tests included**:
- Individual agent unit tests
- Agent tool usage correctness
- Semantic topic and news relevance
- Aget to agent integration tests
- Orchestrator pipeline verification

### 🛡️ Guardrails
- Guardrails-AI used in newsletter_writer.py
- Ensures output quality, safety, non-toxicity, structure
- Rejects hallucinated or toxic markdowns before save