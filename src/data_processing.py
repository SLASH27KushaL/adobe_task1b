# src/data_processing.py

import os
import fitz
import pymupdf4llm
from typing import List, Dict


def extract_sections_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extracts sections and subsections from a PDF using pymupdf4llm Markdown.
    Falls back to splitting full text into paragraph chunks if no headings.
    Returns list of dicts: title, text, page, subsections.
    """
    # Try structured Markdown extraction
    try:
        md = pymupdf4llm.to_markdown(pdf_path)
        lines = md.splitlines()
    except Exception:
        lines = []

    sections: List[Dict] = []
    current_sec: Dict = {'title': None, 'text': '', 'page': None, 'subsections': []}
    current_sub: Dict = None

    # Parse headings (# section, ## subsection)
    for line in lines:
        if line.startswith('# '):
            if current_sec['title']:
                if current_sub:
                    current_sec['subsections'].append(current_sub)
                    current_sub = None
                sections.append(current_sec)
            current_sec = {'title': line[2:].strip(), 'text': '', 'page': None, 'subsections': []}
        elif line.startswith('## '):
            if current_sub:
                current_sec['subsections'].append(current_sub)
            current_sub = {'title': line[3:].strip(), 'text': '', 'page': None}
        else:
            target = current_sub if current_sub else current_sec
            target['text'] += line + '\n'

    # Append last parsed section
    if current_sec.get('title'):
        if current_sub:
            current_sec['subsections'].append(current_sub)
        sections.append(current_sec)

    # Open PDF for fallback & page mapping
    doc = fitz.open(pdf_path)

    # Fallback: if no headings parsed, split full text into paragraph chunks
    if not sections:
        full_text = ''
        for page in doc:
            full_text += page.get_text('text') + '\n'
        # Split into paragraphs by double newlines
        paras = [p.strip() for p in full_text.split('\n\n') if p.strip()]
        # Create single section with paragraph subsections
        fallback_sec = {
            'title': os.path.basename(pdf_path),
            'text': '',
            'page': 0,
            'subsections': []
        }
        for i, para in enumerate(paras, 1):
            fallback_sec['subsections'].append({
                'title': f"Paragraph {i}",
                'text': para,
                'page': 0
            })
        return [fallback_sec]

    # Map section titles to page numbers
    for sec in sections:
        for page in doc:
            if sec['title'] and page.search_for(sec['title']):
                sec['page'] = page.number
                break
        for sub in sec['subsections']:
            for page in doc:
                if sub['title'] and page.search_for(sub['title']):
                    sub['page'] = page.number
                    break

    return sections


def extract_sections_from_folder(folder_path: str) -> List[Dict]:
    """
    Process all PDF files in a folder and extract sections.
    Annotates each section with its source document filename.
    """
    all_secs: List[Dict] = []
    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith('.pdf'):
            continue
        path = os.path.join(folder_path, fname)
        secs = extract_sections_from_pdf(path)
        for sec in secs:
            sec['document'] = fname
        all_secs.extend(secs)
    return all_secs
