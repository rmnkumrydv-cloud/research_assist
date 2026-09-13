"""
embeddings.py — Embedding Model Wrapper (Gate 4)

Initializes HuggingFace embedding models for vector indexing.
"""

import sys
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Get initialized HuggingFaceEmbeddings instance.
    
    Default model: sentence-transformers/all-MiniLM-L6-v2 (fast, 384 dim, local)
    Alternative: BAAI/bge-small-en-v1.5
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=model_name)
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=model_name)
        except ImportError:
            raise ImportError("Neither langchain_huggingface nor langchain_community is installed.")
