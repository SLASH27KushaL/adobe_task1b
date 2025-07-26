import sys
if sys.platform == "win32":
    import types
    sys.modules["resource"] = types.SimpleNamespace()

import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# A minimal English stopword set
STOPWORDS = {
    "the","a","an","and","or","is","are","to","in","that","it","of",
    "for","on","with","as","by","this","these","those","was","were",
    "be","been","has","have","had","but","not","from","at","which"
}

def bm25_tokenize(text: str):
    # find words, lowercase them
    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    # filter out stopwords
    return [t for t in tokens if t not in STOPWORDS]

class HybridRetriever:
    def __init__(self, sections):
        self.sections = sections
        # prepare raw texts
        self.texts = [sec['title'] + "\n" + sec['text'] for sec in sections]
        # BM25 index
        tokenized = [bm25_tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)
        # Semantic index
        self.model = SentenceTransformer("Alibaba-NLP/gte-base-en-v1.5",
                                         trust_remote_code=True)
        self.embeds = self.model.encode(self.texts, normalize_embeddings=True)

    def retrieve(self, persona, job, top_k=20, rrf_k=60):
        query = persona + " " + job

        # semantic scores
        q_vec = self.model.encode([query], normalize_embeddings=True)[0]
        sims = np.dot(self.embeds, q_vec)
        dense_rank = np.argsort(-sims)

        # BM25 scores
        q_tokens = bm25_tokenize(query)
        bm25_scores = self.bm25.get_scores(q_tokens)
        bm25_rank = np.argsort(-bm25_scores)

        # Reciprocal Rank Fusion
        scores = {}
        for i, idx in enumerate(dense_rank, 1):
            scores[idx] = scores.get(idx, 0.0) + 1.0/(i + rrf_k)
        for i, idx in enumerate(bm25_rank, 1):
            scores[idx] = scores.get(idx, 0.0) + 1.0/(i + rrf_k)

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return fused[:top_k]
