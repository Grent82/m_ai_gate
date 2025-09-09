from typing import Optional, Set, Tuple
from core.agent_action import AgentAction, Event
from memory.long_term_memory import LongTermMemory
from memory.short_term_memory import ShortTermMemory
from memory.spatial_memory import SpatialMemory
from core.logger import setup_logger

logger = setup_logger(__name__)

class Agent:
    def __init__(self, name: str, age: int, traits: str, lifestyle: str, position):
        self.name = name
        self.age = age
        self.traits = traits
        self.position = position
        self.lifestyle = lifestyle

        # Profile/identity state (editable by identity revision)
        self.background = (
            "Grew up in a small village and moved to Town Square to explore urban life"
        )
        self.status = "energetic, slightly hungry"

        self.vision_range = 5  # todo
        self.attention_bandwidth: int = 3  # Max number of events agent can attend to
        self.retention = 5  # todo

        self.importance_triggers: Set[str] = set()

        self.current_chat_event: Optional[Event] = None
        self.current_chat_text: Optional[str] = None

        self.short_term_memory = ShortTermMemory()
        self.long_term_memory = LongTermMemory()
        self.spatial_memory = SpatialMemory()

    def add_long_term_memory(self, msg: str, relevance: float):
        self.long_term_memory.add_event(
            Event(subject="", description=msg), relevance=relevance
        )

    def get_recent_event_tuples(self, limit: int = 10) -> Set[Tuple[str, str, str]]:
        return {
            (e.event.subject, e.event.predicate, e.event.object)
            for e in self.long_term_memory.node_sequences["event"][:limit]
        }

    def get_state(self):
        # Collect a small set of recent memory descriptions for prompts
        recent_memories = []
        for t in ("thought", "event", "chat"):
            for node in self.long_term_memory.node_sequences[t][:6]:
                if node.event and node.event.description:
                    recent_memories.append(node.event.description)

        # Provide a lightweight location string as coordinates for prompts
        location = f"({self.position[0]}, {self.position[1]})"

        return {
            "name": self.name,
            "age": self.age,
            "traits": self.traits,
            "lifestyle": self.lifestyle,
            "location": location,
            "memories": recent_memories,
            "background": self.background,
            "status": self.status,
        }
