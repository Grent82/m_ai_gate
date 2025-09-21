import hashlib
from typing import List


def _bytes_to_uint32_list(data: bytes, count: int) -> List[float]:
    out: List[float] = []
    buf = data * ((count * 4) // len(data) + 1)
    for i in range(count):
        start = i * 4
        chunk = buf[start : start + 4]
        val = int.from_bytes(chunk, byteorder="little", signed=False)
        out.append(float(val))
    return out


def get_embedding(text: str, dim: int = 128) -> List[float]:
    """
    Returns a pseudo-embedding of fixed dimension based on text hash.
    Not semantically meaningful — for testing only. No numpy required.
    """
    h = hashlib.sha256((text or "").encode("utf-8")).digest()
    vec = _bytes_to_uint32_list(h, dim)
    norm = sum(v * v for v in vec) ** 0.5
    if norm == 0.0:
        return [0.0] * dim
    return [v / norm for v in vec]
