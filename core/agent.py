from typing import Optional, Set, Tuple, List, Union
from core.agent_action import AgentAction, Event
from memory.long_term_memory import LongTermMemory
from memory.short_term_memory import ShortTermMemory
from memory.spatial_memory import SpatialMemory
from core.logger import setup_logger

logger = setup_logger(__name__)

class Agent:
    def __init__(
        self,
        name: str,
        age: int,
        traits: Union[str, List[str]],
        lifestyle: str,
        position,
        background: Optional[str] = None,
        status: Optional[str] = None,
        sex: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        learned: Optional[str] = None,
        living_area: Optional[str] = None,
    ):
        self.name = name
        self.age = age
        # Store traits as original string and a normalized list for prompts
        self.traits = traits if isinstance(traits, str) else ", ".join(traits)
        self._traits_list: List[str] = (
            [t.strip() for t in traits.split(",") if t.strip()] if isinstance(traits, str) else [t.strip() for t in traits]
        )
        self.position = position
        self.lifestyle = lifestyle

        self.background = background or ""
        self.status = status or ""
        self.sex = (sex or "").lower()

        self.first_name = first_name
        self.last_name = last_name

        self.learned = learned or ""
        self.living_area = living_area or ""

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

        location = f"({self.position[0]}, {self.position[1]})"

        return {
            "name": self.name,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "age": self.age,
            "traits": self.traits,
            "traits_list": self._traits_list,
            "lifestyle": self.lifestyle,
            "learned": self.learned,
            "location": location,
            "memories": recent_memories,
            "background": self.background,
            "status": self.status,
            "sex": self.sex,
            "living_area": self.living_area,
        }
