# src/persona_context.py

def build_context_string(persona: str, job: str) -> str:
    """
    Combine the persona and the job-to-be-done into
    one coherent query string for the retriever.
    """
    # You can customize this template as needed
    return f"Persona: {persona}. Task: {job}"
