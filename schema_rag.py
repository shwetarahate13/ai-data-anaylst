"""Schema + glossary retrieval (RAG).

TF-IDF over schema/glossary chunks keeps the project dependency-light and fully offline.
To upgrade to dense embeddings, replace `_vectorize` / `retrieve` with a sentence-transformers
or Voyage/OpenAI embedding + vector store; the interface stays the same.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from .knowledge import chunks

_SUFFIXES = ("ations", "ation", "ings", "ing", "ied", "ies", "ed", "es", "ly", "s")


def _stem(w: str) -> str:
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def _tokens(text: str) -> list[str]:
    """Lowercase, drop stop words, light suffix stemming ('churned' -> 'churn', 'orders' -> 'order')."""
    return [_stem(w) for w in re.findall(r"[a-z][a-z_]+", text.lower()) if w not in ENGLISH_STOP_WORDS]


class SchemaRAG:
    def __init__(self, docs: list[dict] | None = None):
        self.docs = docs or chunks()
        self.vec = TfidfVectorizer(tokenizer=_tokens, token_pattern=None, lowercase=False, ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vec.fit_transform([d["text"] for d in self.docs])

    def retrieve(self, query: str, k: int = 6) -> list[dict]:
        q = self.vec.transform([query])
        scores = (self.matrix @ q.T).toarray().ravel()
        order = np.argsort(-scores)[:k]
        return [{**self.docs[i], "score": float(scores[i])} for i in order if scores[i] > 0]

    def context(self, query: str, k: int = 6) -> str:
        return "\n".join(f"- {d['text']}" for d in self.retrieve(query, k))
