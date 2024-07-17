import random
import asyncio
from typing import List, Tuple
import torch
from math import ceil

from .prompt import prompt as clean_prompt
from .schema import create_response_model
from ..scorer import Scorer, ScorerInput
from ...features import Example
from ...clients.client import Client
from dataclasses import dataclass

import json


@dataclass
class Sample:
    text: str
    quantile: int
    ground_truth: bool
    predicted: bool = None

    @staticmethod
    def _prepare_samples(
        examples: List, 
        quantile: int,
        ground_truth: bool, 
        tokenizer,
    ):
        samples = []

        for example in examples:
            example.decode(tokenizer)

            samples.append(
                Sample(
                    text=example.text,
                    quantile=quantile,
                    ground_truth = ground_truth
                )
            )

        return samples
    
    def default(self):
        return {
            "text": self.text,
            "quantile": self.quantile,
            "ground_truth": self.ground_truth,
            "predicted": self.predicted,
        }

class RecallScorer(Scorer):
    name = "neighbor"

    def __init__(
        self, 
        client: Client, 
        tokenizer,
        echo: bool = False, 
        temperature: float = 0.0,
        max_tokens: int = 2,
        batch_size: int = 1,
    ):
        self.client = client
        self.tokenizer = tokenizer
        self.echo = echo

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size

    async def __call__(
        self, 
        scorer_in: ScorerInput,
    ) -> List[Sample]:

        samples = self._prepare(
            scorer_in.test_examples,
            scorer_in.record.random_examples
        )

        # Generate responses
        results = await self.process_batches(
            samples,
            scorer_in.explanation
        )

        return results
    
    def _prepare(self, activating_examples, incorrect_examples):

        samples = Sample._prepare_samples(
            incorrect_examples,
            -1,
            False,
            self.tokenizer
        )

        for i, examples in enumerate(activating_examples):

            samples.extend(
                Sample._prepare_samples(
                    examples,
                    i + 1,
                    True,
                    self.tokenizer
                )
            )
        
        return [
            samples[i:i + self.batch_size] 
            for i in range(0, len(samples), self.batch_size)
        ]
    
    async def process_batches(
        self, 
        batches: List[List[Sample]], 
        explanation: str
    ) -> List[Sample]:
        # Create a list of tasks to be executed concurrently
        tasks = [
            self.query(batch, explanation) 
            for batch in batches
        ]

        # Execute the tasks concurrently
        results = await asyncio.gather(*tasks)

        # Return a flattened list of samples
        return [
            item.default()
            for sublist in results 
            for item in sublist
        ]

    def build_prompt(
        self, 
        batch: List[Sample], 
        explanation: str
    ) -> str:
        examples = "\n".join(
            f"Example {i}: {sample.text}" 
            for i, sample in enumerate(batch)
        )

        return clean_prompt(
            explanation=explanation,
            examples=examples,
        )

    async def query(
        self, 
        batch: List[Sample], 
        explanation: str
    ) -> List[Sample]:
        
        batched = len(batch) > 1
        prompt = self.build_prompt(batch, explanation, batched)

        generation_kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        if batched:
            selections = await self.client.generate(
                prompt,
                schema=create_response_model(len(batch)),
                **generation_kwargs
            )

            for i, sample in enumerate(batch):
                sample.predicted = selections[f"example_{i}"] == 1

        else:
            selections = await self.client.generate(
                prompt,
                **generation_kwargs
            )

            batch[0].predicted = int(selections[-1]) == 1

        return batch