from typing import List
from dataclasses import dataclass
from torchtyping import TensorType


@dataclass
class Example:
    tokens: List[int]
    activations: List[float]
    
    def __hash__(self) -> int:
        return hash(tuple(
            self.tokens.tolist()
        ))

    def __eq__(self, other) -> bool:
        return self.tokens == other.tokens
    
    @property
    def max_activation(self):
        return max(self.activations)
    
    @property
    def text(self):
        return "".join(self.str_toks)
    

def load_examples(
    tokens: TensorType["batch", "seq"], 
    activations: TensorType["batch", "seq"]
):
    return [
        Example(
            tokens=toks,
            activations=acts,
        )
        for toks, acts in zip(
            tokens, 
            activations
        )
    ]
    