"""
Phase 3 -- Hybrid RAG Indexing, Retrieval (BM25 + Sentence Transformers) and Reranking
Optimized for in-memory snippet chunking and main.py orchestration.
"""
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class Reranker:
    """Cross-Encoder model for precision reranking of retrieved code snippets."""
    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[tuple[dict, float]], top_k: int = 3) -> list[tuple[dict, float]]:
        if not candidates:
            return []
        pairs = [[query, c["text"]] for c, _ in candidates]
        scores = self.model.predict(pairs)
        scored_candidates = sorted(zip([c for c, _ in candidates], scores), key=lambda x: x[1], reverse=True)
        return scored_candidates[:top_k]


class HybridIndex:
    """Combines Lexical (BM25) and Dense Semantic search with RRF for in-memory chunks."""
    def __init__(self, embedder):
        self.embedder = embedder
        self.chunks: list[dict] = []
        self.tokenized_corpus: list[list[str]] = []
        self.bm25: BM25Okapi | None = None
        self.embeddings = None

    def build(self, chunks: list[dict]):
        """Indexes the provided list of chunk dictionaries in memory."""
        self.chunks = chunks
        if not chunks:
            return
        corpus_texts = [c["text"] for c in chunks]
        self.tokenized_corpus = [text.lower().split() for text in corpus_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        # Encode embeddings using the shared embedder instance passed from main.py
        self.embeddings = self.embedder.encode(corpus_texts, convert_to_tensor=False)


def hybrid_search(
    index: HybridIndex,
    reranker: Reranker,
    query: str,
    retrieve_k: int = 8,
    final_k: int = 3
) -> list[tuple[dict, float]]:
    """
    Executes hybrid search combining BM25 and Dense embeddings via RRF,
    then reranks top candidates using a Cross-Encoder.
    """
    if not index.chunks or not index.bm25:
        return []

    # 1. Lexical BM25 search
    tokenized_query = query.lower().split()
    bm25_scores = index.bm25.get_scores(tokenized_query)
    bm25_ranking = np.argsort(bm25_scores)[::-1]

    # 2. Dense Semantic search
    query_embedding = index.embedder.encode(query, convert_to_tensor=False)
    cos_scores = np.dot(index.embeddings, query_embedding) / (
        np.linalg.norm(index.embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-10
    )
    dense_ranking = np.argsort(cos_scores)[::-1]

    # 3. Reciprocal Rank Fusion (RRF)
    fused_scores = {}
    k = 60
    for rank, idx in enumerate(bm25_ranking):
        fused_scores[idx] = fused_scores.get(idx, 0) + 1 / (k + rank + 1)
    for rank, idx in enumerate(dense_ranking):
        fused_scores[idx] = fused_scores.get(idx, 0) + 1 / (k + rank + 1)

    sorted_indices = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
    
    # Select top initial candidates for reranking
    candidates = [(index.chunks[idx], fused_scores[idx]) for idx in sorted_indices[:retrieve_k]]

    # 4. Cross-Encoder Reranking
    return reranker.rerank(query, candidates, top_k=final_k)