"""
llm.py — LLM Generation via Groq (Gate 6 & Side-Pipelines)

Provides initialized ChatGroq instances and helper functions for LLM operations.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

from langchain_groq import ChatGroq


def get_groq_llm(model_name: str = "qwen/qwen3.8-27b", temperature: float = 0.0, max_tokens: int = 1024) -> ChatGroq:
    """
    Get initialized ChatGroq LLM instance.
    Supports local .env and Streamlit Community Cloud secrets.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            if "GROQ_API_KEY" in st.secrets:
                api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass

    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in environment variables or Streamlit secrets.")

    return ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key
    )


def summarize_table_html(html_str: str, llm: ChatGroq = None) -> str:
    """
    Generate an LLM summary for a table given its HTML string representation.
    """
    if not llm:
        llm = get_groq_llm()

    truncated_html = html_str[:3000] if len(html_str) > 3000 else html_str

    prompt = f"""You are a research paper assistant. Summarize the key data, metrics, comparisons, and findings presented in the following HTML table. Keep the summary concise (2-4 sentences) and highlight exact numerical results or key conclusions.

HTML Table:
{truncated_html}

Summary:"""
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"[WARNING] Table summary LLM call failed: {e}")
        return "Table showing experimental results and comparisons from the paper."


def summarize_image_caption(caption_or_filename: str, page_number: int, llm: ChatGroq = None) -> str:
    """
    Generate a summary/description for an image figure based on caption or page context.
    """
    if not llm:
        llm = get_groq_llm()

    prompt = f"""You are a research paper assistant. Write a concise description (1-2 sentences) of a figure/diagram from page {page_number} of a research paper with context: '{caption_or_filename}'. Explain what this figure likely demonstrates in the context of the paper's architecture or experiments.

Description:"""
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"[WARNING] Image summary LLM call failed: {e}")
        return f"Figure/Diagram on page {page_number} depicting model architecture or experimental findings."
