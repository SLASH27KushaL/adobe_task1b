import os
import json
from typing import List, Dict
import fitz  # PyMuPDF

Section = Dict[str, any]

def load_outline_json(json_path: str) -> List[Section]:
    """
    Read a flat outline JSON (list of {level, text, page})
    and build nested Section objects:
      - Each H1 becomes a top-level section
      - H2s attach under the most recent H1
      - H3s attach under the most recent H2
    """
    data = json.load(open(json_path, encoding="utf-8"))
    flat = data.get("outline", [])
    sections: List[Section] = []

    for entry in flat:
        lvl = entry.get("level")
        txt = entry.get("text", "").strip()
        pg = entry.get("page")

        if lvl == "H1":
            sections.append({
                "document": os.path.basename(json_path).replace("_outline.json", ".pdf"),
                "heading": txt,
                "level": lvl,
                "page": pg,
                "text": "",            # to fill from PDF
                "subsections": []
            })
        elif lvl == "H2" and sections:
            sections[-1]["subsections"].append({
                "document": sections[-1]["document"],
                "heading": txt,
                "level": lvl,
                "page": pg,
                "text": "",
                "subsections": []
            })
        elif lvl == "H3" and sections and sections[-1]["subsections"]:
            sections[-1]["subsections"][-1]["subsections"].append({
                "document": sections[-1]["document"],
                "heading": txt,
                "level": lvl,
                "page": pg,
                "text": ""
            })

    return sections


def extract_sections_from_folder(folder_path: str) -> List[Section]:
    """
    For every *_outline.json in folder_path/docs:
      1. Load the nested outline via load_outline_json()
      2. Open the matching PDF and extract each page’s full text
         into the corresponding section/subsection.
    Returns a flat list of all sections (with subsections nested).
    """
    all_secs: List[Section] = []
    docs_dir = os.path.join(folder_path, "docs")

    for fname in sorted(os.listdir(docs_dir)):
        if not fname.endswith("_outline.json"):
            continue

        outline_path = os.path.join(docs_dir, fname)
        sections = load_outline_json(outline_path)

        # Open PDF once per outline
        pdf_name = fname.replace("_outline.json", ".pdf")
        pdf_path = os.path.join(docs_dir, pdf_name)
        pdf = fitz.open(pdf_path)

        # Fill in text for each section & subsection
        for sec in sections:
            try:
                sec["text"] = pdf[sec["page"] - 1].get_text("text")
            except Exception:
                sec["text"] = ""
            for sub in sec["subsections"]:
                try:
                    sub["text"] = pdf[sub["page"] - 1].get_text("text")
                except Exception:
                    sub["text"] = ""
                # H3 nesting can be handled similarly if needed

        all_secs.extend(sections)

    return all_secs
