from abc import ABC, abstractmethod

class IExecutor(ABC):
    @abstractmethod
    def execute(self, decision: str) -> str:
        pass
