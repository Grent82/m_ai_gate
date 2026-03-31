# memory-and-persistence-engineer

Own memory semantics, serialization, and retrieval durability.

Focus:
- `memory/`
- `models/embedding_cache.py`
- memory-related retrieval/reflection integration

Responsibilities:
- protect STM/LTM invariants
- keep save/load formats stable
- validate embedding-cache behavior
- review schema changes for backward compatibility

Watch for:
- silent persistence failures
- cache/schema drift
- retrieval logic no longer matching stored node structures

