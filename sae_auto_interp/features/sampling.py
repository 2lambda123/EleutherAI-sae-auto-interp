import random
from ..logger import logger

def split_activation_quantiles(examples, n_quantiles):
    max_activation = examples[0].max_activation
    
    thresholds = [max_activation * i / n_quantiles for i in range(1, n_quantiles)]
    quantiles = [[] for _ in range(n_quantiles)]

    for example in examples:
        for i, threshold in enumerate(thresholds):
            if example.max_activation <= threshold:
                quantiles[i].append(example)
                break
        else:
            quantiles[-1].append(example)
    
    return quantiles


def split_quantiles(examples, n_quantiles):
    n = len(examples)
    quantile_size = n // n_quantiles
    
    return [
        examples[i * quantile_size:(i + 1) * quantile_size] 
        for i in range(n_quantiles)
    ]

def check_quantile(quantile, n_test):
    if len(quantile) < n_test:
        logger.error(f"Quantile has too few examples")
        raise ValueError(f"Quantile has too few examples")

def default_sampler(
    record, 
    n_train = 10, 
    n_test = 20,
    **kwargs
):  
    n_samples = n_train + n_test
    samples = random.sample(record.examples, n_samples)
    record.train = samples[:n_train]
    record.test = samples[n_test:]

def sample_activation_quantiles(
    record,
    n_train=10,
    n_test=20,
    n_quantiles=4,
    seed=22,
    **kwargs
):
    random.seed(seed)

    activation_quantiles = split_activation_quantiles(record.examples, n_quantiles)
    train_examples = random.sample(activation_quantiles[0], n_train)

    test_quantiles = activation_quantiles[1:]
    test_examples = []

    per_quantile = n_test // (n_quantiles - 1)
    for quantile in test_quantiles:
        check_quantile(quantile, per_quantile)
        test_examples.append(random.sample(quantile, per_quantile))

    record.train = train_examples
    record.test = test_examples


def sample_top_and_activation_quantiles(
    record,
    n_train=10,
    n_test=20,
    n_quantiles=4,
    seed=22,
    **kwargs
):
    random.seed(seed)
    
    train_examples = record.examples[:n_train]

    activation_quantiles = split_activation_quantiles(
        record.examples[n_train:], n_quantiles
    )

    test_examples = []

    per_quantile = n_test // n_quantiles
    for quantile in activation_quantiles:
        check_quantile(quantile, per_quantile)
        test_examples.append(random.sample(quantile, per_quantile))

    record.train = train_examples
    record.test = test_examples

def sample_top_and_quantiles(
    record,
    n_train: int = 10,
    n_test: int = 20,
    n_quantiles: int = 4,
    seed: int = 22,
    **kwargs
):
    random.seed(seed)

    examples = record.examples

    # Sample n_train examples for training
    train_examples = examples[:n_train]
    remaining_examples = examples[n_train:]

    quantiles = split_quantiles(remaining_examples, n_quantiles)

    test_examples = []

    per_quantile = n_test // n_quantiles
    for quantile in quantiles:
        check_quantile(quantile, per_quantile)
        test_examples.append(random.sample(quantile, per_quantile))

    record.train = train_examples
    record.test = test_examples