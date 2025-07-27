# Dockerfile for persona-driven document intelligence pipeline

# 1. Use a slim, AMD64 base
FROM --platform=linux/amd64 python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Download NLTK punkt model for sentence tokenization
RUN python - <<EOF
import nltk
nltk.download('punkt')
EOF

# 5. Copy source code
COPY src/ ./src
COPY main.py .

# 6. Entrypoint: specify input folder and output JSON path
ENTRYPOINT ["python", "main.py", \
            "--input", "/app/input", \
            "--output", "/app/output/output.json"]
