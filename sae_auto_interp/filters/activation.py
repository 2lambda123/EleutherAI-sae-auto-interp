from .stats import Stat
import numpy as np
from scipy.stats import kurtosis, skew
from typing import List, Dict, Callable, Any
from ..features import Example

def nonzero(example: Example) -> int:
    return np.count_nonzero(example.activations)

def top_tok(example: Example) -> str:
    return example.str_toks[np.argmax(example.activations)]

def top_act(example: Example) -> float:
    return example.max_activation

def apply(functions, array, average=True):
    function_results = {
        function.__name__: [] for function in functions
    }

    for item in array:
        for function in functions:
            function_results[function.__name__].append(function(item))

    for func, result in function_results.items():
        if average:
            function_results[func] = np.mean(result)
        else:
            function_results[func] = result

    return function_results

class Unigram(Stat):
    def __init__(
            self, 
            k: int = 10, 
            nonzero: bool = False,
            top_tok: bool = False,
            top_act: bool = False,
            **kwargs
        ):
        self.k = k
        self.functions = []

        if nonzero:
            self.functions.append(nonzero)
        if top_tok:
            self.functions.append(top_tok)
        if top_act:
            self.functions.append(top_act)

    def refresh(self, k: int = None, **kwargs):
        if k is not None:
            self.k = k
            
    def compute(self, records, *args, **kwargs):
        for record in records:
            self._compute(record, *args, **kwargs)
    
    def _compute(self, record):
        top_k_examples = record.examples[:self.k]
        results = apply(self.functions, top_k_examples)

        for key, value in results.items():
            setattr(record, key, value)

