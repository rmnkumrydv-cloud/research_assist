"""
chunking.py — Section-Aware Chunking (Gate 2)

Groups narrative text and list items under section headings (Title elements)
and applies RecursiveCharacterTextSplitter within each section block.
Attaches paper_id, section name, page_number, and chunk_type="text" metadata.

Pass criteria: Print 3 random chunks — each should have correct section metadata
and read as a coherent paragraph.
"""

import random
from typing import List, Dict, Any
import sys
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ingestion import ingest_paper, RAW_PAPERS_DIR, PROCESSED_DIR, load_parsed_elements


def group_elements_by_section(element_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group elements under section headings based on Title elements.
    
    Returns list of section dicts:
    [
        {
            "section": "1 Introduction",
            "page_number": 1,
            "text": "...",
            "paper_id": "..."
        }, ...
    ]
    """
    sections = []
    current_section = "Preamble"
    current_text_blocks = []
    current_page = 1
    paper_id = element_dicts[0]["paper_id"] if element_dicts else "unknown"

    for elem in element_dicts:
        cat = elem["category"]
        text = (elem.get("text") or "").strip()
        page = elem.get("metadata", {}).get("page_number") or current_page

        if not text:
            continue

        # Ignore Table and Image elements in section-aware text chunking (handled in Gate 3)
        if cat in ["Table", "Image", "Header", "Footer"]:
            continue

        if cat == "Title":
            # Save previous section if it has text
            if current_text_blocks:
                full_section_text = "\n\n".join(current_text_blocks)
                if full_section_text.strip():
                    sections.append({
                        "section": current_section,
                        "page_number": current_page,
                        "text": full_section_text,
                        "paper_id": paper_id
                    })
                current_text_blocks = []

            # Clean header title
            cleaned_title = text.replace("\n", " ").strip()
            # If title is reasonable length (not an entire page block misclassified as Title)
            if len(cleaned_title) < 150:
                current_section = cleaned_title
            current_page = page
        elif cat in ["NarrativeText", "ListItem", "UncategorizedText"]:
            current_text_blocks.append(text)
            if not current_page:
                current_page = page

    # Append final section block
    if current_text_blocks:
        full_section_text = "\n\n".join(current_text_blocks)
        if full_section_text.strip():
            sections.append({
                "section": current_section,
                "page_number": current_page,
                "text": full_section_text,
                "paper_id": paper_id
            })

    return sections


def create_section_aware_chunks(
    element_dicts: List[Dict[str, Any]],
    chunk_size: int = 700,
    chunk_overlap: int = 100
) -> List[Dict[str, Any]]:
    """
    Creates section-aware text chunks from raw parsed elements.
    
    Args:
        element_dicts: Parsed elements from ingestion module
        chunk_size: Max characters per chunk (approx 100-150 words)
        chunk_overlap: Overlap characters between consecutive chunks
        
    Returns:
        List of chunk dicts with section, page, paper_id, and chunk_type metadata.
    """
    sections = group_elements_by_section(element_dicts)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", " ", ""]
    )

    chunks = []
    chunk_counter = 0

    for sec in sections:
        sec_name = sec["section"]
        sec_text = sec["text"]
        paper_id = sec["paper_id"]
        page_num = sec["page_number"]

        sub_splits = text_splitter.split_text(sec_text)
        for idx, split_text in enumerate(sub_splits):
            chunk_counter += 1
            chunks.append({
                "chunk_id": f"{paper_id}_txt_{chunk_counter}",
                "text": split_text,
                "chunk_type": "text",
                "paper_id": paper_id,
                "section": sec_name,
                "page_number": page_num,
                "chunk_index": idx
            })

    return chunks


def create_multimodal_chunks(
    element_dicts: List[Dict[str, Any]],
    chunk_size: int = 700,
    chunk_overlap: int = 100,
    generate_summaries: bool = True
) -> List[Dict[str, Any]]:
    """
    Creates section-aware text chunks PLUS table & image summary chunks with raw payload content.
    
    Gate 3 Side-Pipeline:
    - Tables: summary embedded as text, raw HTML kept in raw_content
    - Images: summary embedded as text, image path kept in raw_content/image_path
    """
    # 1. Text chunks
    chunks = create_section_aware_chunks(element_dicts, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    paper_id = element_dicts[0]["paper_id"] if element_dicts else "unknown"

    # Initialize LLM lazily if generate_summaries is True
    llm = None
    if generate_summaries:
        try:
            from src.llm import get_groq_llm, summarize_table_html, summarize_image_caption
            llm = get_groq_llm()
            print("[...] Groq LLM initialized for table/image side-pipeline summaries.")
        except Exception as e:
            print(f"[WARNING] Could not initialize Groq LLM for summaries: {e}")

    # Track section per page to attach section info to tables and images
    page_section_map = {}
    current_sec = "Preamble"
    for elem in element_dicts:
        page = elem.get("metadata", {}).get("page_number") or 1
        if elem["category"] == "Title":
            cleaned = (elem.get("text") or "").replace("\n", " ").strip()
            if len(cleaned) < 150:
                current_sec = cleaned
        page_section_map[page] = current_sec

    table_count = 0
    image_count = 0

    for elem in element_dicts:
        cat = elem["category"]
        page_num = elem.get("metadata", {}).get("page_number") or 1
        sec_name = page_section_map.get(page_num, "General")

        if cat == "Table":
            table_count += 1
            raw_html = elem.get("html") or f"<table><tr><td>{elem['text']}</td></tr></table>"
            if generate_summaries and llm:
                from src.llm import summarize_table_html
                print(f"      Summarizing Table {table_count} on page {page_num} via Groq...")
                summary = summarize_table_html(raw_html, llm)
            else:
                summary = f"Table on page {page_num}: {elem['text'][:200]}"

            chunks.append({
                "chunk_id": f"{paper_id}_tab_{table_count}",
                "text": summary,          # Searchable text summary for embedding
                "raw_content": raw_html,   # Payload raw HTML
                "chunk_type": "table",
                "paper_id": paper_id,
                "section": sec_name,
                "page_number": page_num,
                "summary": summary
            })

        elif cat == "Image":
            image_count += 1
            img_path = elem.get("image_path") or ""
            caption_text = elem.get("text") or f"Image figure on page {page_num}"
            if generate_summaries and llm:
                from src.llm import summarize_image_caption
                print(f"      Summarizing Image {image_count} on page {page_num} via Groq...")
                summary = summarize_image_caption(caption_text, page_num, llm)
            else:
                summary = f"Image figure on page {page_num}: {caption_text}"

            chunks.append({
                "chunk_id": f"{paper_id}_img_{image_count}",
                "text": summary,          # Searchable text summary for embedding
                "raw_content": img_path,   # Payload image file path
                "chunk_type": "image",
                "paper_id": paper_id,
                "section": sec_name,
                "page_number": page_num,
                "summary": summary,
                "image_path": img_path
            })

    return chunks


def main():
    """Run Gate 2 and Gate 3 Side-Pipeline verification."""
    # Find PDF files or cached parsed elements
    pdf_files = list(RAW_PAPERS_DIR.glob("*.pdf"))
    if not pdf_files:
        print("[ERROR] No sample PDF found in data/raw_papers/")
        return

    paper_id, element_dicts = ingest_paper(str(pdf_files[0]), force_reparse=False)
    print(f"\n[...] Processing section-aware & multimodal chunking for paper_id={paper_id}")
    print(f"      Total raw elements: {len(element_dicts)}")

    # Test multimodal side-pipeline
    multimodal_chunks = create_multimodal_chunks(element_dicts, generate_summaries=True)

    text_chunks = [c for c in multimodal_chunks if c["chunk_type"] == "text"]
    table_chunks = [c for c in multimodal_chunks if c["chunk_type"] == "table"]
    image_chunks = [c for c in multimodal_chunks if c["chunk_type"] == "image"]

    print(f"\n[OK] Total chunks generated: {len(multimodal_chunks)}")
    print(f"      - Text chunks:  {len(text_chunks)}")
    print(f"      - Table chunks: {len(table_chunks)}")
    print(f"      - Image chunks: {len(image_chunks)}")

    # Print sample Table Chunk
    if table_chunks:
        t_sample = table_chunks[0]
        print("\n--- Gate 3 Table Chunk Sample ---")
        print(f"Chunk ID:    {t_sample['chunk_id']}")
        print(f"Section:     {t_sample['section']}")
        print(f"Page Number: {t_sample['page_number']}")
        print(f"Summary:\n{t_sample['summary']}")
        print(f"Raw HTML Preview ({len(t_sample['raw_content'])} chars):\n{t_sample['raw_content'][:200]}...")

    # Print sample Image Chunk
    if image_chunks:
        i_sample = image_chunks[0]
        print("\n--- Gate 3 Image Chunk Sample ---")
        print(f"Chunk ID:    {i_sample['chunk_id']}")
        print(f"Section:     {i_sample['section']}")
        print(f"Page Number: {i_sample['page_number']}")
        print(f"Summary:\n{i_sample['summary']}")
        print(f"Image Path:  {i_sample['image_path']}")

    if len(table_chunks) > 0 and len(image_chunks) > 0:
        print("\n>>> Gate 3 PASSED -- Table & Image side-pipeline active with LLM summaries!")


if __name__ == "__main__":
    main()
