import json
import time
from .data_processing import extract_sections_from_folder
from .ranking import HybridRetriever

def run_pipeline(persona, job, docs_folder):
    sections = extract_sections_from_folder(docs_folder)
    if not sections:
        raise ValueError("No sections found")
    
    retriever = HybridRetriever(sections)
    results = retriever.retrieve(persona, job, top_k=len(sections))

    out = {
        "metadata": {
            "documents": sorted(set(sec['document'] for sec in sections)),
            "persona": persona,
            "job": job,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "sections": []
    }

    for rank, (idx, score) in enumerate(results, 1):
        s = sections[idx]
        entry = {
            "document": s["document"],
            "page": s["page"],
            "title": s["title"],
            "importance_rank": rank,
            "subsections": []
        }
        for sub_rank, sub in enumerate(s.get("subsections", []), 1):
            entry["subsections"].append({
                "document": s["document"],
                "page": sub["page"],
                "title": sub["title"],
                "refined_text": sub["text"].strip(),
                "importance_rank": sub_rank
            })
        out["sections"].append(entry)
    return out

def save_output(output, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
