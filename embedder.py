"""
embedder.py
-----------
Hybrid feature representation layer.
Combines TF-IDF sparse vectors with dense neural embeddings
(sentence-transformers / CareerBERT-style) for richer semantic matching.
"""

import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# Use a lightweight but effective sentence-transformer model
# Falls back to TF-IDF only if torch/transformers not available
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 80MB, fast, strong semantics
MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

try:
    from sentence_transformers import SentenceTransformer
    _st_model = SentenceTransformer(EMBEDDING_MODEL, cache_folder=MODEL_CACHE_DIR)
    TRANSFORMER_AVAILABLE = True
except Exception:
    _st_model = None
    TRANSFORMER_AVAILABLE = False
    print("[Embedder] sentence-transformers unavailable — using TF-IDF only.")


class HybridEmbedder:
    """
    Produces hybrid embeddings:
      - TF-IDF component: captures lexical/keyword relevance
      - Neural component: captures contextual/semantic similarity
    Final vector = [tfidf_vector | neural_vector] (concatenated, L2-normalized)
    """

    def __init__(self, tfidf_max_features: int = 5000, alpha: float = 0.4):
        """
        alpha: weight of TF-IDF vs neural in final vector.
               0.4 means 40% TF-IDF, 60% neural.
        """
        self.alpha = alpha
        self.tfidf = TfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.tfidf_fitted = False

    def fit(self, corpus: list[str]):
        """Fit TF-IDF on a corpus of job + resume texts."""
        self.tfidf.fit(corpus)
        self.tfidf_fitted = True

    def _tfidf_embed(self, texts: list[str]) -> np.ndarray:
        if not self.tfidf_fitted:
            raise RuntimeError("Call fit() before embedding.")
        matrix = self.tfidf.transform(texts).toarray()
        return normalize(matrix, norm="l2")

    def _neural_embed(self, texts: list[str]) -> np.ndarray:
        if not TRANSFORMER_AVAILABLE or _st_model is None:
            # Return zero vectors if model unavailable
            return np.zeros((len(texts), 384))
        embeddings = _st_model.encode(texts, show_progress_bar=False, batch_size=32)
        return normalize(np.array(embeddings), norm="l2")

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Produce hybrid embeddings for a list of texts.
        Returns numpy array of shape (n_texts, tfidf_dim + neural_dim).
        """
        tfidf_vecs = self._tfidf_embed(texts)
        neural_vecs = self._neural_embed(texts)

        # Weighted concatenation
        combined = np.hstack([
            self.alpha * tfidf_vecs,
            (1 - self.alpha) * neural_vecs,
        ])
        return normalize(combined, norm="l2")

    def embed_single(self, text: str) -> np.ndarray:
        """Convenience method for a single document."""
        return self.embed([text])[0]

    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalized vectors."""
        return float(np.dot(vec_a, vec_b))

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"tfidf": self.tfidf, "tfidf_fitted": self.tfidf_fitted,
                         "alpha": self.alpha}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.tfidf = state["tfidf"]
        self.tfidf_fitted = state["tfidf_fitted"]
        self.alpha = state["alpha"]


# Singleton
embedder = HybridEmbedder()
