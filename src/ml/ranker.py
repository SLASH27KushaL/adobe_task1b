# src/ml/ranker.py

import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple, Dict

# Minimal stopword set for BM25 tokenization
STOPWORDS = {
    "the","a","an","and","or","is","are","to","in","that","it","of",
    "for","on","with","as","by","this","these","those","was","were",
    "be","been","has","have","had","but","not","from","at","which"
}

def bm25_tokenize(text: str) -> List[str]:
    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return [t for t in tokens if t not in STOPWORDS]

class HybridRetriever:
    def __init__(self, sections: List[Dict]):
        """
        sections: list of dicts each having keys:
          - 'heading', 'text'
        """
        self.sections = sections
        # Build combined text blobs for each section
        self.texts = [sec["heading"] + "\n" + sec["text"] for sec in sections]

        # BM25 index
        tokenized_corpus = [bm25_tokenize(txt) for txt in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Semantic index (sentence-transformers)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeds = self.model.encode(
            self.texts, convert_to_tensor=True, normalize_embeddings=True
        )

    def retrieve(
        self,
        context: str,
        top_k: int = 5,
        rrf_k: int = 60
    ) -> List[Tuple[int, float]]:
        """
        Returns a list of (section_index, fused_score) sorted descending.
        """
        # 1) Semantic similarity
        q_emb = self.model.encode(
            context, convert_to_tensor=True, normalize_embeddings=True
        )
        sims = util.pytorch_cos_sim(q_emb, self.embeds)[0].cpu().numpy()
        dense_rank = np.argsort(-sims)

        # 2) BM25 scores
        bm25_scores = self.bm25.get_scores(bm25_tokenize(context))
        bm25_rank = np.argsort(-bm25_scores)

        # 3) Reciprocal Rank Fusion
        scores: Dict[int, float] = {}
        for i, idx in enumerate(dense_rank, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0/(i + rrf_k)
        for i, idx in enumerate(bm25_rank, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0/(i + rrf_k)

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return fused[:top_k]

    def refine_subsections(
        self,
        text: str,
        top_k_sentences: int = 5
    ) -> str:
        """
        Splits 'text' into sentences, embeds them, and
        returns the top_k_sentences most central sentences.
        """
        from nltk import sent_tokenize

        sentences = sent_tokenize(text)
        if not sentences:
            return ""

        # Embed sentences and full chunk
        sent_embs = self.model.encode(
            sentences, convert_to_tensor=True, normalize_embeddings=True
        )
        chunk_emb = self.model.encode(
            text, convert_to_tensor=True, normalize_embeddings=True
        )

        # Compute similarity and pick top sentences
        sims = util.pytorch_cos_sim(chunk_emb, sent_embs)[0]
        top_indices = sims.topk(top_k_sentences).indices.cpu().numpy().tolist()
        # Return in original order
        return " ".join([sentences[i] for i in sorted(top_indices)])
