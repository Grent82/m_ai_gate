from abc import ABC, abstractmethod
from typing import List

from core.agent import Agent
from memory.long_term_memory import MemoryNode


class IReflector(ABC):
    @abstractmethod
    def reflect(self, agent: Agent) -> List[MemoryNode]:
        """Run reflection for the given agent and return created thought nodes."""
        raise NotImplementedError
