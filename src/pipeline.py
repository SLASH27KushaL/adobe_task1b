# src/pipeline.py

import os
import json
from typing import Dict, List
from .data_processing import extract_sections_from_folder, Section
from .persona_context import build_context_string
from .ml.ranker import HybridRetriever
from .utils import timestamp

def run_pipeline(
    input_folder: str,
    persona: str,
    job: str,
    top_k: int = 5
) -> Dict:
    """
    1. Extract sections (with subsections) from input/docs/
    2. Build persona+job context string
    3. Retrieve top_k sections via HybridRetriever
    4. Refine each section and its subsections
    5. Assemble final JSON structure
    """
    # 1. Extraction
    sections: List[Section] = extract_sections_from_folder(input_folder)
    if not sections:
        raise RuntimeError("No sections found in input/docs")

    # 2. Build retrieval context
    context = build_context_string(persona, job)

    # 3. Initialize retriever & get top sections
    retriever = HybridRetriever(sections)
    top_results = retriever.retrieve(context, top_k=top_k)

    # 4. Assemble output
    output = {
        "metadata": {
            "documents": sorted({sec["document"] for sec in sections}),
            "persona": persona,
            "job_to_be_done": job,
            "timestamp": timestamp()
        },
        "sections": []
    }

    for rank, (idx, score) in enumerate(top_results, start=1):
        sec = sections[idx]
        entry = {
            "document": sec["document"],
            "page": sec["page"],
            "section_title": sec["heading"],
            "importance_rank": rank,
            "refined_text": retriever.refine_subsections(sec["text"]),
            "subsections": []
        }

        # refine nested subsections
        for sub_rank, sub in enumerate(sec.get("subsections", []), start=1):
            entry["subsections"].append({
                "document": sub["document"],
                "page": sub["page"],
                "section_title": sub["heading"],
                "importance_rank": sub_rank,
                "refined_text": retriever.refine_subsections(sub["text"])
            })

        output["sections"].append(entry)

    return output

def save_output(output: Dict, out_path: str):
    """
    Write the output JSON to disk, creating directories as needed.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
