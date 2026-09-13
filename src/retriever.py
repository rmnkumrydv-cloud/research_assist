"""
retriever.py — Vector Store & Advanced Retrieval Pipeline (Gate 4 & Gate 5)

Features:
- Vector Store Indexing (Chroma / Qdrant) with MultiVector payload metadata
- Hybrid Search: Dense Vector Similarity + Sparse BM25 Keyword Search
- Query Rewriting & Decomposition via Groq LLM
- Reranking / Reciprocal Rank Fusion (RRF) for retrieval precision
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from src.embeddings import get_embedding_model
from src.llm import get_groq_llm

VECTOR_DB_DIR = PROJECT_ROOT / "data" / "processed" / "chroma_db"


def chunks_to_documents(chunks: List[Dict[str, Any]], paper_name: str = "") -> List[Document]:
    """Convert chunk dicts into LangChain Document instances with full metadata."""
    docs = []
    for c in chunks:
        doc = Document(
            page_content=c["text"],
            metadata={
                "chunk_id": c.get("chunk_id", ""),
                "paper_id": c.get("paper_id", ""),
                "paper_name": c.get("paper_name", paper_name),
                "section": c.get("section", "General"),
                "page_number": c.get("page_number", 1),
                "chunk_type": c.get("chunk_type", "text"),
                "raw_content": c.get("raw_content", c["text"]),
                "image_path": c.get("image_path", ""),
                "summary": c.get("summary", "")
            }
        )
        docs.append(doc)
    return docs


class AdvancedRetriever:
    """
    Advanced Multimodal RAG Retriever implementing:
    - Dense Vector Search (Chroma)
    - Sparse BM25 Keyword Search
    - Query Decomposition
    - Reciprocal Rank Fusion (RRF) Reranking
    """

    def __init__(
        self,
        chunks: List[Dict[str, Any]] = None,
        collection_name: str = "research_papers",
        persist_dir: Optional[Path] = VECTOR_DB_DIR
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir or VECTOR_DB_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_model = get_embedding_model()
        self.vectorstore = None
        self.bm25_retriever = None

        if chunks:
            self.index_chunks(chunks)

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """Index chunks into both Chroma Vector Store and BM25 Retriever."""
        from langchain_community.vectorstores import Chroma
        from langchain_community.retrievers import BM25Retriever

        docs = chunks_to_documents(chunks)
        print(f"\n[...] Indexing {len(docs)} document chunks into Chroma vector store...")

        # Initialize Chroma vector store
        self.vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embedding_model,
            collection_name=self.collection_name,
            persist_directory=str(self.persist_dir)
        )
        print(f"[OK] Indexed {len(docs)} documents in Chroma ({self.persist_dir})")

        # Initialize BM25 retriever for sparse keyword matching
        print("[...] Initializing BM25 sparse keyword retriever...")
        self.bm25_retriever = BM25Retriever.from_documents(docs)
        self.bm25_retriever.k = 5

    def decompose_query(self, query: str) -> List[str]:
        """
        Gate 5: Query Decomposition / Rewriting via Groq LLM.
        Breaks multi-part questions into sub-queries.
        """
        try:
            llm = get_groq_llm(temperature=0.0)
            prompt = f"""You are a research query planner. If the user query contains multiple sub-questions or complex aspects, break it down into 2-3 specific sub-queries. If the query is simple, return just the original query.

User Query: "{query}"

Output ONLY the sub-queries, one per line:"""
            response = llm.invoke(prompt)
            lines = [line.strip("- *123. ").strip() for line in response.content.split("\n") if line.strip()]
            queries = [q for q in lines if len(q) > 3]
            if query not in queries:
                queries.insert(0, query)
            return queries[:3]
        except Exception as e:
            print(f"[WARNING] Query decomposition failed: {e}")
            return [query]

    def _reciprocal_rank_fusion(self, results_list: List[List[Document]], k: int = 60) -> List[Document]:
        """
        Reciprocal Rank Fusion (RRF) to merge and rerank multiple document result lists.
        RRF_score(doc) = sum(1 / (k + rank_i))
        """
        doc_scores = {}
        doc_map = {}

        for docs in results_list:
            for rank, doc in enumerate(docs, start=1):
                doc_id = doc.metadata.get("chunk_id") or doc.page_content[:50]
                doc_map[doc_id] = doc
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                doc_scores[doc_id] += 1.0 / (k + rank)

        # Sort by RRF score descending
        sorted_doc_ids = sorted(doc_scores.keys(), key=lambda d: doc_scores[d], reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_doc_ids]

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Gate 5: Hybrid Search combining Dense (Vector) + Sparse (BM25) with RRF Reranking.
        """
        if not self.vectorstore:
            raise ValueError("Vector store is not initialized. Call index_chunks first.")

        # 1. Dense vector search
        dense_docs = self.vectorstore.similarity_search(query, k=top_k)

        # 2. Sparse BM25 search
        sparse_docs = []
        if self.bm25_retriever:
            self.bm25_retriever.k = top_k
            sparse_docs = self.bm25_retriever.invoke(query)

        # 3. Reciprocal Rank Fusion
        merged_docs = self._reciprocal_rank_fusion([dense_docs, sparse_docs])
        return merged_docs[:top_k]

    def advanced_search(self, query: str, top_k: int = 5, use_query_decomp: bool = True) -> List[Document]:
        """
        Full Gate 5 Advanced Search Pipeline:
        Query Decomposition -> Hybrid Search for each sub-query -> RRF Reranking -> Top-K
        """
        if use_query_decomp:
            sub_queries = self.decompose_query(query)
            print(f"[...] Query Decomposition generated sub-queries: {sub_queries}")
        else:
            sub_queries = [query]

        sub_results = []
        for q in sub_queries:
            docs = self.hybrid_search(q, top_k=top_k)
            sub_results.append(docs)

        final_reranked = self._reciprocal_rank_fusion(sub_results)
        return final_reranked[:top_k]


class MultiPaperRetriever:
    """
    Multi-Paper Retriever supporting indexing and cross-paper search
    across multiple AdvancedRetriever instances using Reciprocal Rank Fusion (RRF).
    """
    def __init__(self):
        self.retrievers: Dict[str, AdvancedRetriever] = {}
        self.paper_names: Dict[str, str] = {}
        self.paper_chunks: Dict[str, List[Dict[str, Any]]] = {}

    def add_paper(
        self,
        paper_id: str,
        paper_name: str,
        chunks: List[Dict[str, Any]],
        retriever: Optional[AdvancedRetriever] = None
    ) -> AdvancedRetriever:
        """Register or index a paper retriever."""
        if retriever is None:
            retriever = AdvancedRetriever(chunks=chunks, collection_name=f"paper_{paper_id}")
        self.retrievers[paper_id] = retriever
        self.paper_names[paper_id] = paper_name
        self.paper_chunks[paper_id] = chunks
        return retriever

    def get_paper_chunks(self, paper_id: str) -> List[Dict[str, Any]]:
        """Return the chunks for a given paper."""
        return self.paper_chunks.get(paper_id, [])

    def get_all_papers(self) -> Dict[str, Dict[str, Any]]:
        """Return metadata about all registered papers."""
        return {
            pid: {
                "name": self.paper_names.get(pid, pid),
                "chunk_count": len(self.paper_chunks.get(pid, []))
            }
            for pid in self.retrievers
        }

    def remove_paper(self, paper_id: str):
        """Remove a paper from the retriever registry."""
        if paper_id in self.retrievers:
            del self.retrievers[paper_id]
        if paper_id in self.paper_names:
            del self.paper_names[paper_id]
        if paper_id in self.paper_chunks:
            del self.paper_chunks[paper_id]

    def advanced_search(
        self,
        query: str,
        top_k: int = 5,
        use_query_decomp: bool = True,
        paper_id: Optional[str] = None
    ) -> List[Document]:
        """
        Perform advanced search across a specific paper or all registered papers.
        Uses RRF fusion to merge rankings across papers when querying all.
        """
        if not self.retrievers:
            return []

        # If specific paper selected and exists
        if paper_id and paper_id != "all":
            if paper_id in self.retrievers:
                docs = self.retrievers[paper_id].advanced_search(
                    query, top_k=top_k, use_query_decomp=use_query_decomp
                )
                for d in docs:
                    if "paper_name" not in d.metadata or not d.metadata["paper_name"]:
                        d.metadata["paper_name"] = self.paper_names.get(paper_id, paper_id)
                    d.metadata["paper_id"] = paper_id
                return docs
            return []

        # Query all papers and merge using RRF
        paper_results = []
        for pid, ret in self.retrievers.items():
            docs = ret.advanced_search(query, top_k=top_k, use_query_decomp=use_query_decomp)
            for d in docs:
                d.metadata["paper_name"] = self.paper_names.get(pid, pid)
                d.metadata["paper_id"] = pid
            if docs:
                paper_results.append(docs)

        if not paper_results:
            return []
        if len(paper_results) == 1:
            return paper_results[0][:top_k]

        # Reciprocal Rank Fusion across papers
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        k = 60
        for doc_list in paper_results:
            for rank, doc in enumerate(doc_list, start=1):
                uid = f"{doc.metadata.get('paper_id')}_{doc.metadata.get('chunk_id') or doc.page_content[:60]}"
                doc_map[uid] = doc
                if uid not in doc_scores:
                    doc_scores[uid] = 0.0
                doc_scores[uid] += 1.0 / (k + rank)

        sorted_uids = sorted(doc_scores.keys(), key=lambda u: doc_scores[u], reverse=True)
        return [doc_map[u] for u in sorted_uids][:top_k]


def main():
    """Run Gate 4 & Gate 5 verification on sample paper."""
    from src.ingestion import ingest_paper, RAW_PAPERS_DIR
    from src.chunking import create_multimodal_chunks

    pdf_files = list(RAW_PAPERS_DIR.glob("*.pdf"))
    if not pdf_files:
        print("[ERROR] No sample PDF found.")
        return

    # Ingest & Chunk
    paper_id, element_dicts = ingest_paper(str(pdf_files[0]), force_reparse=False)
    chunks = create_multimodal_chunks(element_dicts, generate_summaries=False) # Fast summaries for testing

    print(f"\n[...] Building AdvancedRetriever for paper_id={paper_id}...")
    retriever = AdvancedRetriever(chunks=chunks, collection_name=f"paper_{paper_id}")

    # Test Query 1: Mixed chunk types
    test_query = "What is the Transformer model architecture and how does Table 1 compare performance?"
    print(f"\n[...] Running Advanced Hybrid Search query: '{test_query}'")

    results = retriever.advanced_search(test_query, top_k=5)

    print("\n--- Top Retrieved Results ---")
    chunk_types_found = set()
    for i, doc in enumerate(results, 1):
        c_type = doc.metadata.get("chunk_type")
        chunk_types_found.add(c_type)
        print(f"\n[{i}] Chunk Type: {c_type.upper()} | Section: {doc.metadata.get('section')} | Page: {doc.metadata.get('page_number')}")
        print(f"    Summary/Content: {doc.page_content[:150]}...")
        if c_type == "table":
            print(f"    Raw HTML Present: {bool(doc.metadata.get('raw_content'))}")
        elif c_type == "image":
            print(f"    Image Path: {doc.metadata.get('image_path')}")

    print(f"\n--- Gate 4 & Gate 5 Check ---")
    print(f"  Retrieved chunk types: {chunk_types_found}")

    if len(results) > 0 and len(chunk_types_found) > 1:
        print("\n>>> Gate 4 & Gate 5 PASSED -- Hybrid Search, RRF Reranking, and MultiVector Payload active!")


if __name__ == "__main__":
    main()
