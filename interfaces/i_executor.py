from abc import ABC, abstractmethod
from core.agent import Agent
from core.world import World


class IExecutor(ABC):
    @abstractmethod
    def execute(self, agent: Agent, world: World, max_steps: int = 1) -> str:
        """Advance the agent along its current plan and return a status string."""
        pass
