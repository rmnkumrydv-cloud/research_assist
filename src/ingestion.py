"""
ingestion.py — Document Ingestion & Parsing (Gate 1)

Loads raw PDF research papers using Unstructured.io and separates them
into text, table, and image elements. Saves parsed elements to JSON
for reuse without re-parsing.

Pass criteria: Running this on a sample paper prints a count of
text/table/image elements found, and it's non-zero for all three.
"""

import json
import os
import sys
import hashlib
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_PAPERS_DIR = PROJECT_ROOT / "data" / "raw_papers"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def generate_paper_id(pdf_path: str) -> str:
    """Generate a unique paper ID from the filename."""
    filename = Path(pdf_path).stem
    # Create a short hash for uniqueness
    file_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
    # Clean filename for use as ID
    clean_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in filename)
    return f"{clean_name}_{file_hash}"

class FallbackElement:
    def __init__(self, element_id: str, category: str, text: str, page_number: int):
        self.id = element_id
        self.category = category
        self.text = text
        self.metadata = FallbackMetadata(page_number)
        
    def __str__(self):
        return self.text


class FallbackMetadata:
    def __init__(self, page_number: int):
        self.page_number = page_number
        self.coordinates = None
        self.parent_id = None


def parse_pdf_with_pymupdf_fallback(pdf_path: str) -> list:
    """
    Robust fallback parser using PyMuPDF (fitz) to extract text blocks when Unstructured/OpenCV is unavailable.
    """
    doc = fitz.open(pdf_path)
    elements = []
    elem_idx = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        text_blocks = page.get_text("blocks")
        for b in text_blocks:
            block_text = b[4].strip()
            if not block_text:
                continue
            
            # Simple category heuristic
            category = "NarrativeText"
            if len(block_text.splitlines()) == 1 and len(block_text) < 100 and block_text[0].isupper():
                category = "Title"
            
            elements.append(FallbackElement(
                element_id=f"pymupdf_{page_num+1}_{elem_idx}",
                category=category,
                text=block_text,
                page_number=page_num + 1
            ))
            elem_idx += 1

    print(f"      PyMuPDF Fallback elements extracted: {len(elements)}")
    return elements


def load_and_parse_pdf(pdf_path: str, strategy: str = "fast") -> list:
    """
    Parse a PDF file into structured elements using Unstructured.io (with PyMuPDF fallback).
    
    Args:
        pdf_path: Path to the PDF file
        strategy: Parsing strategy - 'fast' for speed, 'hi_res' for unstructured layout
    
    Returns:
        List of element objects with content, category, and metadata
    """
    print(f"\n[...] Parsing PDF: {Path(pdf_path).name}")
    print(f"      Strategy: {strategy}")

    # Ensure image output dir exists
    image_dir = PROCESSED_DIR / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    try:
        from unstructured.partition.pdf import partition_pdf
        elements = partition_pdf(
            filename=pdf_path,
            strategy=strategy,
        )
        print(f"      Raw elements extracted: {len(elements)}")
        return elements
    except (ImportError, Exception) as e:
        print(f"[WARNING] Unstructured partition unavailable/failed ({type(e).__name__}: {e}). Using PyMuPDF text parser fallback...")
        return parse_pdf_with_pymupdf_fallback(pdf_path)



def elements_to_dicts(elements: list, paper_id: str) -> list:
    """
    Convert Unstructured elements to serializable dicts with metadata.
    
    Each element dict contains:
    - element_id: unique ID
    - category: NarrativeText, Table, Image, Title, etc.
    - text: the text content
    - metadata: page_number, coordinates, etc.
    - paper_id: identifier for the source paper
    """
    element_dicts = []

    for elem in elements:
        elem_dict = {
            "element_id": elem.id,
            "category": elem.category,
            "text": str(elem),
            "paper_id": paper_id,
            "metadata": {
                "page_number": elem.metadata.page_number if hasattr(elem.metadata, "page_number") else None,
                "coordinates": str(elem.metadata.coordinates) if hasattr(elem.metadata, "coordinates") and elem.metadata.coordinates else None,
                "parent_id": elem.metadata.parent_id if hasattr(elem.metadata, "parent_id") else None,
            }
        }

        # For tables, also store the HTML representation
        if elem.category == "Table":
            elem_dict["html"] = elem.metadata.text_as_html if hasattr(elem.metadata, "text_as_html") else None

        # For images, store the image path if available
        if elem.category == "Image":
            elem_dict["image_path"] = elem.metadata.image_path if hasattr(elem.metadata, "image_path") else None

        element_dicts.append(elem_dict)

    return element_dicts


def save_parsed_elements(element_dicts: list, paper_id: str) -> str:
    """Save parsed elements to JSON for reuse without re-parsing."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / f"{paper_id}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(element_dicts, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved {len(element_dicts)} elements to {output_path}")
    return str(output_path)


def load_parsed_elements(paper_id: str) -> list:
    """Load previously parsed elements from JSON."""
    json_path = PROCESSED_DIR / f"{paper_id}.json"
    if not json_path.exists():
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_element_summary(element_dicts: list):
    """Print a summary of element categories found."""
    categories = Counter(e["category"] for e in element_dicts)

    print("\n--- Element Summary ---")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    print(f"  TOTAL: {len(element_dicts)}")

    # Check pass criteria
    text_count = categories.get("NarrativeText", 0)
    table_count = categories.get("Table", 0)
    image_count = categories.get("Image", 0)

    print(f"\n--- Gate 1 Check ---")
    print(f"  NarrativeText elements: {text_count} {'[OK]' if text_count > 0 else '[MISSING]'}")
    print(f"  Table elements:         {table_count} {'[OK]' if table_count > 0 else '[MISSING]'}")
    print(f"  Image elements:         {image_count} {'[OK]' if image_count > 0 else '[MISSING]'}")

    if text_count > 0 and table_count > 0 and image_count > 0:
        print(f"\n>>> Gate 1 PASSED -- All element types detected!")
    else:
        missing = []
        if text_count == 0:
            missing.append("NarrativeText")
        if table_count == 0:
            missing.append("Table")
        if image_count == 0:
            missing.append("Image")
        print(f"\n[WARNING] Missing element types: {', '.join(missing)}")
        print("  This might be a strategy or dependency issue (poppler, tesseract).")
        print("  Or the sample paper may not contain tables/figures.")


def extract_tables_with_pdfplumber(pdf_path: str, paper_id: str) -> list:
    """Extract tables using pdfplumber and format as HTML tables."""
    import pdfplumber

    table_elements = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for t_idx, table_data in enumerate(tables, start=1):
                if not table_data or len(table_data) < 1:
                    continue

                # Format table as HTML
                html_lines = ["<table border='1'>"]
                rows_text = []
                for row_idx, row in enumerate(table_data):
                    html_lines.append("  <tr>")
                    tag = "th" if row_idx == 0 else "td"
                    row_cells = []
                    for cell in row:
                        cell_text = (cell or "").strip().replace("\n", " ")
                        html_lines.append(f"    <{tag}>{cell_text}</{tag}>")
                        if cell_text:
                            row_cells.append(cell_text)
                    html_lines.append("  </tr>")
                    if row_cells:
                        rows_text.append(" | ".join(row_cells))
                html_lines.append("</table>")

                html_str = "\n".join(html_lines)
                plain_text = "\n".join(rows_text)

                if plain_text.strip():
                    table_elements.append({
                        "element_id": f"{paper_id}_p{page_idx}_tab{t_idx}",
                        "category": "Table",
                        "text": plain_text,
                        "html": html_str,
                        "paper_id": paper_id,
                        "metadata": {
                            "page_number": page_idx,
                            "parent_id": None,
                        }
                    })
    return table_elements


def extract_images_with_pymupdf(pdf_path: str, paper_id: str) -> list:
    """Extract embedded images using PyMuPDF (pymupdf)."""
    import pymupdf as fitz

    image_dir = PROCESSED_DIR / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    image_elements = []
    doc = fitz.open(pdf_path)

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list, start=1):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Skip tiny icons/decorations (< 2000 bytes)
                if len(image_bytes) < 2000:
                    continue

                image_filename = f"{paper_id}_p{page_num}_img{img_idx}.{image_ext}"
                image_filepath = image_dir / image_filename

                with open(image_filepath, "wb") as f:
                    f.write(image_bytes)

                image_elements.append({
                    "element_id": f"{paper_id}_p{page_num}_img{img_idx}",
                    "category": "Image",
                    "text": f"Figure/Image from page {page_num} ({image_filename})",
                    "image_path": str(image_filepath),
                    "paper_id": paper_id,
                    "metadata": {
                        "page_number": page_num,
                        "parent_id": None,
                    }
                })
            except Exception as e:
                print(f"[WARNING] Failed to extract image xref {xref} on page {page_num}: {e}")

    doc.close()
    return image_elements


def ingest_paper(pdf_path: str, force_reparse: bool = False) -> tuple:
    """
    Full ingestion pipeline for a single paper.
    Combines Unstructured.io parsing with pdfplumber table extraction
    and PyMuPDF image extraction to ensure high accuracy.
    
    Returns:
        (paper_id, element_dicts)
    """
    paper_id = generate_paper_id(pdf_path)

    # Check for cached parse
    if not force_reparse:
        cached = load_parsed_elements(paper_id)
        if cached:
            print(f"[OK] Using cached parse for paper_id={paper_id}")
            return paper_id, cached

    # 1. Parse text & structure using Unstructured (fast strategy as fallback if hi_res lacks OCR dependencies)
    elements = load_and_parse_pdf(pdf_path)
    element_dicts = elements_to_dicts(elements, paper_id)

    # 2. Check if tables or images are missing from unstructured output
    has_tables = any(e["category"] == "Table" for e in element_dicts)
    has_images = any(e["category"] == "Image" for e in element_dicts)

    if not has_tables:
        print("[...] Extracting tables with pdfplumber...")
        table_elements = extract_tables_with_pdfplumber(pdf_path, paper_id)
        element_dicts.extend(table_elements)
        print(f"      Extracted {len(table_elements)} table(s)")

    if not has_images:
        print("[...] Extracting images with PyMuPDF...")
        image_elements = extract_images_with_pymupdf(pdf_path, paper_id)
        element_dicts.extend(image_elements)
        print(f"      Extracted {len(image_elements)} image(s)")

    # Sort elements by page_number if present
    element_dicts.sort(key=lambda x: x.get("metadata", {}).get("page_number") or 0)

    # Save for reuse
    save_parsed_elements(element_dicts, paper_id)

    return paper_id, element_dicts


def main():
    """Run ingestion on a sample paper from data/raw_papers/."""
    RAW_PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    # Find PDF files
    pdf_files = list(RAW_PAPERS_DIR.glob("*.pdf"))

    if not pdf_files:
        print("[ERROR] No PDF files found in data/raw_papers/")
        print("  Please add a research paper PDF to: " + str(RAW_PAPERS_DIR))
        print("  Tip: Use a paper that has tables AND figures for best Gate 1 results.")
        return

    # Process first paper
    pdf_path = str(pdf_files[0])
    print(f"Processing: {pdf_path}")

    paper_id, element_dicts = ingest_paper(pdf_path, force_reparse=True)

    # Print summary
    print_element_summary(element_dicts)

    # Print a few sample elements
    print("\n--- Sample Elements ---")
    for i, elem in enumerate(element_dicts[:5]):
        text_preview = elem["text"][:100] + "..." if len(elem["text"]) > 100 else elem["text"]
        print(f"  [{i}] {elem['category']}: {text_preview}")


if __name__ == "__main__":
    main()
