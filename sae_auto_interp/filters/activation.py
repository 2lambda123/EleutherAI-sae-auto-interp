from .stats import Stat
import numpy as np
from typing import List
from ..features import Example, FeatureRecord

def nonzero(example: Example) -> int:
    return np.count_nonzero(example.activations)

def top_tok(example: Example) -> str:
    return example.tokens[np.argmax(example.activations)]

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
        nonzero = []
        top_tok = []

        for example in record.examples[:self.k]:
            nonzero.append(nonzero(example))
            top_tok.append(top_tok(example))

        nonzero = np.mean(nonzero)
        top_tok = np.mean(top_tok)

        record.nonzero = float(nonzero)
        record.top_tok = float(top_tok)



