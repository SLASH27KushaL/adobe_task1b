def test_run_pipeline_smoke():
    from src.pipeline import run_pipeline
    out = run_pipeline("AI researcher", "Summarize recent LLM benchmarks", "test_docs/")
    assert "sections" in out
