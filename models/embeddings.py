import hashlib
import numpy as np

def get_embedding(text: str, dim: int = 128) -> list:
    """
    Returns a pseudo-embedding of fixed dimension based on text hash.
    Not semantically meaningful — for testing only.
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    arr = np.frombuffer(h * ((dim * 4) // len(h) + 1), dtype=np.uint32)[:dim]
    arr = arr.astype(np.float32)
    arr = arr / np.linalg.norm(arr)
    return arr.tolist()
