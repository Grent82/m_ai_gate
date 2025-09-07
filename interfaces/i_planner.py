from abc import ABC, abstractmethod
from typing import Dict, List
from core.agent import Agent
from core.world import World
from memory.long_term_memory import MemoryNode


class IPlanner(ABC):
    @abstractmethod
    def plan(self, agent: Agent, world: World, retrieved: Dict[str, Dict[str, List[MemoryNode]]]) -> str:
        pass
