import json
import os
import hashlib
from typing import Dict, List, Optional

from .embeddings import get_embedding


class EmbeddingCache:
    """
    Simple file-backed embedding cache.

    - Uses SHA256(text) as the default key unless a custom key is provided.
    - Persists to `data/embeddings.json` by default.
    """

    def __init__(self, path: str = "data/embeddings.json") -> None:
        self.path = path
        self._data: Dict[str, List[float]] = {}
        self._ensure_dir()
        self._load()

    def _ensure_dir(self) -> None:
        d = os.path.dirname(self.path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    def _load(self) -> None:
        try:
            if os.path.exists(self.path):
                with open(self.path, "r") as f:
                    self._data = json.load(f)
            else:
                self._data = {}
        except Exception:
            self._data = {}

    def _save(self) -> None:
        try:
            with open(self.path, "w") as f:
                json.dump(self._data, f)
        except Exception:
            pass

    @staticmethod
    def key_from_text(text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

    def get(self, text: str, *, key: Optional[str] = None, dim: int = 128) -> List[float]:
        k = key or self.key_from_text(text)
        if k in self._data:
            return self._data[k]
        vec = get_embedding(text, dim=dim)
        self._data[k] = vec
        self._save()
        return vec

    def put(self, key: str, vector: List[float]) -> None:
        self._data[key] = vector
        self._save()

    def get_by_key(self, key: str) -> Optional[List[float]]:
        return self._data.get(key)


_default_cache = EmbeddingCache()


def get_embedding_cached(text: str, *, key: Optional[str] = None, dim: int = 128) -> List[float]:
    return _default_cache.get(text, key=key, dim=dim)


def add_embedding_to_cache(key: str, vector: List[float]) -> None:
    _default_cache.put(key, vector)


def get_embedding_from_cache(key: str) -> Optional[List[float]]:
    return _default_cache.get_by_key(key)

