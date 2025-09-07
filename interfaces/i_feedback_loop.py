from abc import ABC, abstractmethod

class IFeedbackLoop(ABC):
    @abstractmethod
    def adapt(self, reflection: str) -> str:
        pass
