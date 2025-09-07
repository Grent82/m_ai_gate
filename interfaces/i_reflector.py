from abc import ABC, abstractmethod

class IReflector(ABC):
    @abstractmethod
    def reflect(self, execution_result: str) -> str:
        pass
