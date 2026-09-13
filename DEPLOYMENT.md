# Gate 11 — Streamlit Community Cloud Deployment Guide

This guide details how to deploy the **Advanced Research Paper Multimodal RAG System** to Streamlit Community Cloud.

---

## 📋 Prerequisites

1. **GitHub Repository:** Push this codebase (`rag-project`) to your GitHub account.
2. **Groq API Key:** Obtain an API key from [console.groq.com](https://console.groq.com).

---

## 🚀 Step-by-Step Deployment

### 1. Repository Setup
Ensure your repository includes:
- `app.py` (Main Streamlit Entrypoint)
- `requirements.txt` (Pinned dependencies)
- `src/` (Core ingestion, chunking, retrieval, guardrails, and LLM modules)
- `.streamlit/config.toml` (Optional theme configuration)

### 2. Streamlit Community Cloud Setup
1. Go to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
2. Click **New app**.
3. Select your repository, branch (`main`), and set Main file path to `app.py`.

### 3. Add Environment Secrets
1. In the deployment configuration page (or App Settings > Secrets), add your Groq API key and optional LangSmith API key for full RAG observability:
   ```toml
   GROQ_API_KEY = "gsk_your_groq_api_key_here"

   # LangSmith RAG Observability (Optional)
   LANGCHAIN_TRACING_V2 = "true"
   LANGCHAIN_API_KEY = "ls__your_langsmith_api_key_here"
   LANGCHAIN_PROJECT = "research-assist-rag"
   ```
2. Click **Save & Deploy**.

---

## ⚙️ Architecture Highlights in Production

- **Secrets Handling:** `src/llm.py` automatically detects `st.secrets["GROQ_API_KEY"]` when running in the cloud.
- **Multimodal Pipeline:** Uses `PyMuPDF` and `pdfplumber` for zero-C-binary extraction of tables and figures.
- **Vector DB:** Embedded `Chroma` instance runs in-memory or persisted in temporary volume space.
- **Audit Logging:** Guardrail triggers logged live to `data/processed/guardrail_logs.jsonl`.
- **RAGAS Evaluation Scorecard:** Accessible directly in the UI via the **ℹ️ RAG Evaluation** popover header.
