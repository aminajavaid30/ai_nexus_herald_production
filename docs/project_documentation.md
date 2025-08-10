# 📁 AI Nexus Herald: Project Documentation

> A living documentation hub for developers, collaborators, and maintainers of the **AI Nexus Herald** system.

---

## ✅ 1. System Purpose & Context

**AI Nexus Herald** is a multi-agent AI system designed to **automatically generate professional newsletters** on trending topics in the field of Artificial Intelligence.

### 🎯 Purpose

To reduce the manual overhead of discovering, researching, and compiling high-quality AI news into a consistent and engaging newsletter format.

### 👤 Target Users

- AI researchers and professionals
- Technical newsletter publishers
- AI product teams and analysts
- Content curators

### 🧩 Problem Solved

- Manual curation is time-consuming
- Lack of consistent newsletter formats
- Scattered sources of trustworthy AI updates
- Scalability and consistency issues in human-authored digests

---

## ✅ 2. System Architecture

### 🔨 Components Overview

- **Streamlit Frontend**: User-friendly interface to trigger newsletter generation and view outputs
- **FastAPI Backend**: API for orchestrating agents, saving data, and monitoring execution
- **LangGraph Workflow**: Orchestrates the multi-agent system
- **Agents**:
  - `TopicFinderAgent`: Extracts trending AI topics from RSS feeds
  - `DeepResearchAgent`: Gathers contextually rich, factual information about each topic
  - `NewsletterWriterAgent`: Synthesizes a professional, structured markdown newsletter
  - `Orchestrator`: Handles the whole LangGraph orchestartion pipeline 
- **Groq API (LLM)**: Language model backend used by agents for reasoning and content generation
- **Evaluation Suite**: Tests implemented using DeepEval to assess output structure and quality
- **OpenAI API (LLM)**: Required by DeepEval

### 🔄 Data Flow
RSS Feeds --> Topic Finder --> Deep Researcher --> Newsletter Writer --> Markdown Output


### 🧰 Technologies Used

- Python 3.12
- LangChain / LangGraph
- Groq API (LLM)
- FastAPI
- Streamlit
- DeepEval for testing
- Guardrails AI for output safety

---

## ✅ 3. Model Development & Evaluation

### 🧠 Reasoning Strategy

The system follows a **modular agent-based architecture**, where each agent handles a specific cognitive task:

1. **Topic Identification**: Uses semantic similarity and rule-based filters on RSS headlines to select 5 trending topics in AI.
2. **Deep Research**: Constructs reference search queries, retrieves contextual facts, and generates summaries for each topic.
3. **Newsletter Writing**: Assembles topic and facts into a structured, multi-section newsletter using custom prompts.

Each stage builds on the previous, ensuring traceability and fact alignment throughout the pipeline.

---

### 🤖 Agents & Tools

| Agent | Role | Tools Used |
|-------|------|------------|
| `TopicFinderAgent` | Selects relevant and diverse AI topics | RSS Feeds Title Extraction |
| `DeepResearchAgent` | Retrieves supporting information | RSS Feeds News Extraction |
| `NewsletterWriterAgent` | Generates the final newsletter in markdown format | Structured prompt + Guardrails |

- All agents are orchestrated using **LangGraph** for serial execution and state management.
- Prompts are stored in a centralized YAML config (`prompt_config.yaml`).

---

### ✅ Evaluation Strategy

#### 🧪 Unit & Integration Tests

- **Agent-level Tests**:
  - Tool correctness
  - Output structure
  - Guardrails validation

- **Pipeline Tests**:
  - LangGraph orchestrator end-to-end run
  - Output faithfulness, coherence, and semantic relevance using `DeepEval`

#### 🧪 Example Metrics (via DeepEval)

- `GEval` structured scoring (Relevance, Structure, Faithfulness)
- `ToolCorrectness`
- Semantic similarity scores

#### 📋 Manual Review

- Visual inspection of markdown newsletters
- News source validation through provided links
- Prompt tuning through iterative reviews

---

## 🧩 Future Enhancements

- Replace RSS feeds with semantic web search
- Integrate image generation (OpenAI DALL·E or Stability AI)
- Multi-language newsletter support
- Real-time user feedback on newsletter quality
- Scheduled deployment with email automation

---

## 👩‍💻 Repo Structure
```sh
├── 📂 outputs/ # Generated newsletters and logs
├── 📂 resources/ # Static files like logos, images, etc.
├── 📂 src/
│ ├── 📂 backend/
│ │ ├── 📂 agents/
│ │ │ ├── deep_researcher.py # Gathers detailed insights on topics
│ │ │ ├── newsletter_writer.py # Composes newsletter content
│ │ │ ├── orchestrator.py # Coordinates multi-agent flow
│ │ │ └── topic_finder.py # Identifies trending AI topics
│ │ ├── 📂 config/
│ │ │ ├── config.yaml # System and service-level config
│ │ │ └── prompt_config.yaml # Prompt templates for LLM agents
│ │ │ └── rss_config.yaml # RSS feed URLs
│ │ ├── logger.py # Custom logging utility
│ │ ├── main.py # FastAPI backend entry point
│ │ ├── paths.py # Centralized path definitions
│ │ ├── prompt_builder.py # Builds structured prompts for agents
│ │ ├── tools.py # Custom tools for agents
│ │ └── utils.py # Utility functions
│ └── 📂 frontend/
│ ├──── 📂 pages/
│ │ ├──── 1 Newsletter.py # Streamlit UI for newsletter generation
│ ├──── Home.py # Streamlit app entry point
│ └──── style.css # Custom CSS styles
├── 📂 tests/
│ ├─── pytest.ini
│ ├─── test_topics.py
│ ├─── test_news.py
│ ├─── test_newsletter.py
│ ├─── test_topic_finder.py
│ ├─── test_deep_researcher.py
│ ├─── test_newsletter_writer.py
│ ├─── test_orchestrator.py 
├── .env # Environment variables
├── .gitignore # Git ignored files
├── LICENSE # Project license
├── README.md # Project documentation
└── requirements.txt # Python dependencies
```

---

## 📌 Conclusion
This project is designed to be **extensible, safe, and production-ready**. With its modular agent design, clear orchestration logic, and evaluation protocols, **AI Nexus Herald** can be deployed and scaled with confidence.

> Keep this document updated as agents evolve, prompts change, and new features are added.