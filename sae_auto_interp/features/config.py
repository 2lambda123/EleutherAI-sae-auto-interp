from dataclasses import dataclass

@dataclass
class FeatureConfig:

    min_examples: int = 100
    
    max_examples: int = 5_000

    sample: bool = True

    n_train: int = 10

    n_test: int = 20

    n_quantiles: int = 4

    seed: int = 22