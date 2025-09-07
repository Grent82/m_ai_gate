from abc import ABC, abstractmethod

class IPlanner(ABC):
    @abstractmethod
    def plan(self, input_data: str) -> str:
        pass