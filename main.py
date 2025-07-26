# main.py

# Fix for Windows: patch Unix-only `resource` module to avoid crash
import sys
if sys.platform == "win32":
    import types
    sys.modules["resource"] = types.SimpleNamespace()

import argparse
from src.pipeline import run_pipeline, save_output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--docs", required=True)
    parser.add_argument("--output", default="output.json")
    args = parser.parse_args()

    result = run_pipeline(args.persona, args.job, args.docs)
    save_output(result, args.output)
    print(f"✅ Output written to {args.output}")

if __name__ == "__main__":
    main()
