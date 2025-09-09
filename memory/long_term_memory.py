from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from core.agent_action import Event
from core.logger import setup_logger

logger = setup_logger(__name__)


class MemoryNode:
    def __init__(
        self,
        node_id: str,
        node_type: str,
        created: datetime,
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
        node = self.add_memory_node("event", event, relevance, keywords, filling, None, embedding)
        if keywords:
            for keyword in keywords:
                key = keyword.lower()
                self.keyword_to_event.setdefault(key, []).insert(0, node)
                if f"{event.predicate} {event.object}" != "is idle":
                    self.keyword_strength_event[key] = self.keyword_strength_event.get(key, 1) + 1
        return node

    def add_thought(self, event: Event, relevance: float, keywords: Optional[set] = None, filling: List[List[str]] = None, embedding: Optional[List[float]] = None) -> MemoryNode:
        expiration = datetime.now() + timedelta(days=30)
        node = self.add_memory_node("thought", event, relevance, keywords, filling, expiration, embedding)

        if keywords:
            for keyword in keywords:
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
    
    def add_memory_node(self, node_type: str, event: Event, relevance: float, keywords: Optional[set] = None, filling: List[List[str]] = None, expiration: Optional[datetime] = None, embedding: Optional[List[float]] = None) -> MemoryNode:
        node_id = self._generate_node_id()
        node = MemoryNode(
            node_id=node_id,
            node_type=node_type,
            created=datetime.now(),
            expiration=expiration,
            event=event,
            relevance=relevance,
            keywords=keywords,
            filling=filling,
            embedding=embedding
        )
        self.node_sequences[node_type].insert(0, node)
        return node
    
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
