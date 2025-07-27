#!/usr/bin/env python3

import argparse
import json
from src.pipeline import run_pipeline, save_output

def main():
    parser = argparse.ArgumentParser(
        description="Persona-driven document intelligence pipeline"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to root folder containing 'docs/' and 'persona.json'"
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to write the final output JSON"
    )
    args = parser.parse_args()

    # Load persona configuration
    persona_cfg = json.load(open(f"{args.input}/persona.json", encoding="utf-8"))
    persona = persona_cfg["persona"]
    job = persona_cfg["job"]
    top_k = persona_cfg.get("top_k", 5)

    # Run the pipeline
    result = run_pipeline(args.input, persona, job, top_k=top_k)

    # Save the output
    save_output(result, args.output)
    print(f"✅ Output written to {args.output}")

if __name__ == "__main__":
    main()
