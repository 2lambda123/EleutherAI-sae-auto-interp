from typing import List

from ...features import Example
from ...oai_autointerp import (
    ActivationRecord,
    ExplanationNeuronSimulator,
    LogprobFreeExplanationTokenSimulator,
    simulate_and_score,
)
from ..scorer import Scorer, ScorerResult


class OpenAISimulator(Scorer):
    """
    Simple wrapper for the LogProbFreeExplanationTokenSimulator.
    """

    name = "simulator"
<<<<<<< HEAD
    
=======

>>>>>>> d3b81f50d676295bd88d0e499194b44c40c02b13
    def __init__(
        self,
        client,
        tokenizer,
        all_at_once=True,
    ):
        self.client = client
<<<<<<< HEAD
=======
        self.tokenizer = tokenizer  
        self.all_at_once = all_at_once
>>>>>>> d3b81f50d676295bd88d0e499194b44c40c02b13

    async def __call__(self, record):
        # Simulate and score the explanation.
        cls = (
            ExplanationNeuronSimulator
            if self.all_at_once
            else LogprobFreeExplanationTokenSimulator
        )
        simulator = cls(
            self.client,
            record.explanation,
        )

        valid_activation_records = self.to_activation_records(record.test)

        result = await simulate_and_score(simulator, valid_activation_records)

        return ScorerResult(
            record=record,
            score=result,
        )

    def to_activation_records(self, examples: List[Example]) -> List[ActivationRecord]:
        return [
            ActivationRecord(
                self.tokenizer.batch_decode(example.tokens), example.activations
            )
            for example in examples
        ]
