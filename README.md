# Persona-Driven Document Intelligence Solution

This project implements the Persona-Driven Document Intelligence challenge. Given a **persona description**, a **job-to-be-done (JTBD)**, and a set of PDF documents, the system extracts and ranks the most relevant sections and subsections from the documents.

**Key Features:**

- **Hybrid Retrieval:** Combines semantic search (using sentence-transformer embeddings) with keyword search (BM25):contentReference[oaicite:20]{index=20}:contentReference[oaicite:21]{index=21}.
- **Reciprocal Rank Fusion:** Merges the two rankings so that sections important by either method are prioritized:contentReference[oaicite:22]{index=22}.
- **Structured Chunking:** Uses PyMuPDF4LLM to convert PDFs into Markdown, detecting headings by font size:contentReference[oaicite:23]{index=23}. Sections (`#`) and subsections (`##`) are extracted and indexed.
- **Offline CPU Execution:** All processing is done offline on CPU. The Docker build preloads the transformer model and NLTK tokenizer.
- **Fast Performance:** Designed to process 3–5 documents within 60 seconds on a typical CPU by efficient indexing and batch encoding.

## File Structure

- `data_processing.py`: Parses PDFs into sections/subsections.
- `ranking.py`: Builds BM25 and embedding indices; performs hybrid retrieval with RRF.
- `pipeline.py`: Coordinates the end-to-end flow and formats the output JSON.
- `main.py`: Command-line interface for running the pipeline.
- `requirements.txt`: Python dependencies.
- `Dockerfile`: Container setup (pre-downloads models/tokenizers).
- `README.md`: This documentation.
- `tests/`: (Optional) Contains unit tests (if any).

## How It Works

1. **Extract Sections:** Each PDF is converted to Markdown via **PyMuPDF4LLM**. Headings become Markdown headers, which we parse into sections (`#`) and subsections (`##`):contentReference[oaicite:24]{index=24}.
2. **Embed and Index:** Each section’s text is encoded with the **Alibaba GTE** model (`gte-base-en-v1.5`), a 768-dimensional embedding model that supports long documents:contentReference[oaicite:25]{index=25}. Simultaneously, a BM25 index is built over the same text.
3. **Retrieve:** For a query combining the persona and JTBD, we compute cosine similarities to all section embeddings, and BM25 scores to all sections.
4. **Fuse Rankings:** We apply **Reciprocal Rank Fusion** (RRF) to merge the two ranked lists. RRF gives higher weight to sections that rank well in either list:contentReference[oaicite:26]{index=26}.
5. **Output JSON:** Results are output as JSON with metadata and two tiers: ranked sections and, within each, their ranked subsections.

This hybrid approach leverages the precision of keyword matching and the generality of semantic matching:contentReference[oaicite:27]{index=27}:contentReference[oaicite:28]{index=28}. The final JSON includes each section’s document, page, title, and importance rank, plus its subsections (with text excerpts).

## Usage

### Prerequisites

- Python 3.8+ (tested on 3.10).
- Docker (for containerized execution).

### Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
