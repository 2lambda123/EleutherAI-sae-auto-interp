from .stats import Stat
import numpy as np
from typing import List
from ..features import Example, FeatureRecord

def nonzero(example: Example) -> int:
    return np.count_nonzero(example.activations)

def top_tok(example: Example) -> str:
    return example.tokens[np.argmax(example.activations)].item()

class Unigram(Stat):
    def __init__(
            self, 
            k: int = 10
        ):
        self.k = k

    def refresh(self, k: int = None, **kwargs):
        if k is not None:
            self.k = k
            
    def compute(self, records: List[FeatureRecord], **kwargs):
        for record in records:
            self._compute(record)
    
    def _compute(self, record: FeatureRecord):
        avg_nonzero = []
        top_tokens = []

        for example in record.examples[:self.k]:
            avg_nonzero.append(nonzero(example))
            top_tokens.append(top_tok(example))

        record.n_unique = len(set(top_tokens))
        
        avg_nonzero = np.mean(avg_nonzero)
        record.avg_nonzero = float(avg_nonzero)



