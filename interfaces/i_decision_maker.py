from abc import ABC, abstractmethod

class IDecisionMaker(ABC):
    @abstractmethod
    def decide(self, plan: str) -> str:
        pass
