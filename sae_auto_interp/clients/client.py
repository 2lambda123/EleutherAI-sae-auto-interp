from abc import ABC, abstractmethod

class Client(ABC):
    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def generate(self, prompt: str, generation_args: dict) -> str:
        pass
