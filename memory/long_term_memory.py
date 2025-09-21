from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
import os
import json
from core.agent_action import Event
from core.logger import setup_logger
from models.embedding_cache import (
    get_embedding_cached,
    add_embedding_to_cache,
    get_embedding_from_cache,
    EmbeddingCache,
)

logger = setup_logger(__name__)


class MemoryNode:
    def __init__(
        self,
        node_id: str,
        node_type: str,
        created: datetime,
        depth: int,
        relevance: float,
        expiration: Optional[datetime] = None,
        keywords: Optional[set] = None,
        event: Optional[Event] = None,
        embedding: Optional[List[float]] = None,
        filling: Optional[List[List[str]]] = None
    ):
        self.node_id = node_id
        self.node_type = node_type  # "event", "thought", "chat"
        self.created = created
        self.last_accessed = created
        self.depth = depth
        self.relevance = relevance
        self.expiration = expiration
        self.keywords = keywords
        self.embedding = embedding
        self.filling = filling

        self.event = event

    def is_expired(self) -> bool:
        return self.expiration is not None and datetime.now() > self.expiration


class LongTermMemory:
    def __init__(self):
        self.node_sequences: Dict[str, List[MemoryNode]] = {
            "event": [],
            "thought": [],
            "chat": [],
        }
        self.node_counter = 0

        self.keyword_to_event: Dict[str, List[MemoryNode]] = {}
        self.keyword_to_thought: Dict[str, List[MemoryNode]] = {}
        self.keyword_to_chat: Dict[str, List[MemoryNode]] = {}

        self.keyword_strength_event: Dict[str, int] = {}
        self.keyword_strength_thought: Dict[str, int] = {}

    def _generate_node_id(self) -> str:
        self.node_counter += 1
        return f"node_{self.node_counter}"

    def add_event(self, event: Event, relevance: float, keywords: Optional[set] = None, embedding: Optional[List[float]] = None, filling: List[List[str]] = None) -> MemoryNode:
        node = self.add_memory_node("event", event, relevance, keywords, filling, None, embedding, depth=0)
        if keywords:
            for keyword in keywords:
                key = keyword.lower()
                self.keyword_to_event.setdefault(key, []).insert(0, node)
                if f"{event.predicate} {event.object}" != "is idle":
                    self.keyword_strength_event[key] = self.keyword_strength_event.get(key, 1) + 1
        return node

    def add_thought(self, event: Event, relevance: float, keywords: Optional[set] = None, filling: List[List[str]] = None, embedding: Optional[List[float]] = None) -> MemoryNode:
        expiration = datetime.now() + timedelta(days=30)
        base_keys: Set[str] = set()
        for k in (event.subject, event.predicate, event.object):
            if k:
                base_keys.add(str(k))
        all_keys: Set[str] = set(keywords) if keywords else set()
        all_keys |= base_keys

        computed_depth = 1
        try:
            if filling:
                cand_ids = []
                for f in filling:
                    if isinstance(f, str):
                        cand_ids.append(f)
                    elif isinstance(f, list):
                        for x in f:
                            if isinstance(x, str):
                                cand_ids.append(x)
                id_to_node = {}
                for t in ("event", "thought", "chat"):
                    for n in self.node_sequences.get(t, []):
                        id_to_node[getattr(n, "node_id", "")] = n
                depths = [getattr(id_to_node.get(cid, None), "depth", 0) for cid in cand_ids if cid in id_to_node]
                if depths:
                    computed_depth = 1 + max(depths)
        except Exception:
            computed_depth = 1

        node = self.add_memory_node("thought", event, relevance, all_keys or None, filling, expiration, embedding, depth=computed_depth)

        if all_keys:
            for keyword in all_keys:
                key = keyword.lower()
                self.keyword_to_thought.setdefault(key, []).insert(0, node)
                if f"{event.predicate} {event.object}" != "is idle":
                    self.keyword_strength_thought[key] = self.keyword_strength_thought.get(key, 1) + 1
        return node

    def add_chat(
        self,
        event: Event,
        relevance: float,
        keywords: Optional[set] = None,
        filling: List[List[str]] = None,
        embedding: Optional[List[float]] = None,
    ) -> MemoryNode:
        node = self.add_memory_node("chat", event, relevance, keywords, filling, None, embedding)
        if keywords:
            for keyword in keywords:
                key = keyword.lower()
                self.keyword_to_chat.setdefault(key, []).insert(0, node)
        return node
    
    def add_memory_node(self, node_type: str, event: Event, relevance: float, keywords: Optional[set] = None, filling: List[List[str]] = None, expiration: Optional[datetime] = None, embedding: Optional[List[float]] = None, depth: int = 0) -> MemoryNode:
        node_id = self._generate_node_id()
        node = MemoryNode(
            node_id=node_id,
            node_type=node_type,
            created=datetime.now(),
            depth=depth,
            expiration=expiration,
            event=event,
            relevance=relevance,
            keywords=keywords,
            filling=filling,
            embedding=embedding
        )
        self.node_sequences[node_type].insert(0, node)
        return node

    # --- Persistence ---
    def save(self, out_dir: str) -> None:
        """
        Persist LongTermMemory to disk, mirroring the structure of code_examples.

        - nodes.json: nodes with metadata + embedding_key
        - kw_strength.json: keyword strengths for event/thought
        Embeddings themselves are stored in models/embedding_cache.py's cache file.
        """
        os.makedirs(out_dir, exist_ok=True)

        all_nodes: List[MemoryNode] = []
        for t in ("event", "thought", "chat"):
            all_nodes.extend(self.node_sequences.get(t, []))
        all_nodes.sort(key=lambda n: n.created)

        nodes_dict: Dict[str, dict] = {}
        for idx, node in enumerate(all_nodes, start=1):
            node_id = node.node_id or f"node_{idx}"

            created_str = node.created.strftime('%Y-%m-%d %H:%M:%S') if node.created else None
            expiration_str = node.expiration.strftime('%Y-%m-%d %H:%M:%S') if node.expiration else None

            desc = getattr(node.event, "description", "") if getattr(node, "event", None) else ""
            emb_key = EmbeddingCache.key_from_text(desc) if desc else None
            if emb_key and node.embedding:
                add_embedding_to_cache(emb_key, node.embedding)

            nodes_dict[node_id] = {
                "node_count": idx,
                "type_count": None,
                "type": node.node_type,
                "depth": node.depth,
                "created": created_str,
                "expiration": expiration_str,
                "subject": getattr(node.event, "subject", None) if node.event else None,
                "predicate": getattr(node.event, "predicate", None) if node.event else None,
                "object": getattr(node.event, "object", None) if node.event else None,
                "description": desc,
                "embedding_key": emb_key,
                "relevance": node.relevance,
                "keywords": list(node.keywords) if node.keywords else [],
                "filling": node.filling,
            }

        with open(os.path.join(out_dir, "nodes.json"), "w") as f:
            json.dump(nodes_dict, f)

        with open(os.path.join(out_dir, "kw_strength.json"), "w") as f:
            json.dump(
                {
                    "kw_strength_event": self.keyword_strength_event,
                    "kw_strength_thought": self.keyword_strength_thought,
                },
                f,
            )

    @classmethod
    def load(cls, in_dir: str) -> "LongTermMemory":
        ltm = cls()
        nodes_path = os.path.join(in_dir, "nodes.json")
        kw_path = os.path.join(in_dir, "kw_strength.json")
        if not os.path.exists(nodes_path):
            return ltm

        try:
            with open(nodes_path, "r") as f:
                nodes_data = json.load(f)
        except Exception:
            return ltm

        # Load keyword strengths if present
        try:
            if os.path.exists(kw_path):
                with open(kw_path, "r") as f:
                    kw = json.load(f)
                    ltm.keyword_strength_event = kw.get("kw_strength_event", {}) or {}
                    ltm.keyword_strength_thought = kw.get("kw_strength_thought", {}) or {}
        except Exception:
            pass

        # Reconstruct nodes in ascending node_count and prepend to match newest-first ordering
        def node_sort_key(item):
            _, d = item
            try:
                return int(str(d.get("node_count", "0")))
            except Exception:
                return 0

        items = sorted(nodes_data.items(), key=node_sort_key)
        max_numeric_id = 0
        for node_id, d in items:
            try:
                # Parse metadata
                node_type = d.get("type")
                depth = int(d.get("depth", 0))
                created = datetime.strptime(d.get("created"), '%Y-%m-%d %H:%M:%S') if d.get("created") else datetime.now()
                expiration = datetime.strptime(d.get("expiration"), '%Y-%m-%d %H:%M:%S') if d.get("expiration") else None
                relevance = float(d.get("relevance", 1.0))
                keywords = set(d.get("keywords", []) or [])
                filling = d.get("filling", None)

                # Event reconstruction
                event = Event(
                    subject=d.get("subject"),
                    predicate=d.get("predicate"),
                    object=d.get("object"),
                    description=d.get("description") or "",
                )

                # Embedding via key (cache), or compute with caching
                emb_key = d.get("embedding_key")
                embedding = None
                if emb_key:
                    embedding = get_embedding_from_cache(emb_key)
                    if embedding is None:
                        embedding = get_embedding_cached(event.description, key=emb_key)
                elif event.description:
                    embedding = get_embedding_cached(event.description)

                # Create the MemoryNode and insert at front
                node = MemoryNode(
                    node_id=node_id,
                    node_type=node_type,
                    created=created,
                    depth=depth,
                    relevance=relevance,
                    expiration=expiration,
                    keywords=keywords or None,
                    event=event,
                    embedding=embedding,
                    filling=filling,
                )
                ltm.node_sequences.setdefault(node_type, []).insert(0, node)

                # Update keyword indices
                if keywords:
                    for kw in keywords:
                        key = str(kw).lower()
                        if node_type == "event":
                            ltm.keyword_to_event.setdefault(key, []).insert(0, node)
                        elif node_type == "thought":
                            ltm.keyword_to_thought.setdefault(key, []).insert(0, node)
                        elif node_type == "chat":
                            ltm.keyword_to_chat.setdefault(key, []).insert(0, node)

                # Track numeric suffix if any
                try:
                    if node_id.startswith("node_"):
                        n = int(node_id.split("_", 1)[1])
                        max_numeric_id = max(max_numeric_id, n)
                except Exception:
                    pass
            except Exception:
                continue

        # Ensure future node ids continue after the max seen
        ltm.node_counter = max(ltm.node_counter, max_numeric_id)
        return ltm
    
    def get_summarized_latest_events(self, retention:int) -> Set[Event]:
        latest_events = set()
        node_list = list(self.node_sequences["event"])
        for node in node_list[:retention]:
            latest_events.add(node.event)
        return latest_events
    
    def retrieve_relevant_events(self, event: Event) -> List[MemoryNode]:
        logger.info("[Memory] Fast retrieval via keyword index for events...")
        candidates = set()

        for key in [event.subject, event.predicate, event.object]:
            if not key:
                continue
            keyword = key.lower()
            nodes = self.keyword_to_event.get(keyword, [])
            candidates.update(nodes)

        return [node for node in candidates if not node.is_expired()]
    
    def retrieve_relevant_thoughts(self, event: Event) -> List[MemoryNode]:
        logger.info("[Memory] Fast retrieval via keyword index for thoughts...")
        candidates = set()

        for key in [event.subject, event.predicate, event.object]:
            if not key:
                continue
            keyword = key.lower()
            nodes = self.keyword_to_thought.get(keyword, [])
            candidates.update(nodes)

        return [node for node in candidates if not node.is_expired()]
