<<<<<<< HEAD
import os
import json
from tqdm import tqdm
import torch

def get_dims(model, examples):
    return model.tokenizer(
        examples, 
        return_tensors='pt',
        padding=True, 
        truncation=True
    ).input_ids.shape

def _score(model, submodule, examples, features):
    batch_size, seq_len = get_dims(model, examples)
    features = torch.tensor(features).long()
    
    with model.trace(examples):
        scores = submodule.ae.output
        expanded_features = features.unsqueeze(1).expand(batch_size, seq_len)
        activations = scores.gather(
            2, expanded_features.unsqueeze(-1)
        ).squeeze(-1)
        scores = torch.max(activations, dim=1).values
        activations.save()

    return activations.value

def to_feature(string):
    string = string.replace(".txt", "").replace("feature", "") 
    return string.split("_")

def score(model, submodule_dict, examples_dir, batch_size=10):

    example_queue = []
    feature_queue = []
    scores = []

    current_submodule = list(submodule_dict.keys())[0]

    for file in tqdm(os.listdir(examples_dir)):
        with open(os.path.join(examples_dir, file), "r") as f:
            examples = json.load(f)['result']

        module, feature = to_feature(file)

        example_queue.append(examples)
        feature_queue.append(int(feature))

        if (
            module != current_submodule 
            or len(example_queue) >= batch_size
        ):
            
            s = _score(
                model, 
                submodule_dict[current_submodule], 
                sum(example_queue, []),
                feature_queue
            )

            scores.append(s)
            example_queue.clear()
            feature_queue.clear()

            current_submodule = module

    return scores
=======
import json
import os
from typing import Dict, List

import orjson
import torch
from tqdm import tqdm


def score(
    model,
    submodule_dict: Dict,
    examples_dir: str,
    generation_size: int = 10,
    batch_size: int = 10,
):
    counter = 0
    current_module = None
    running_examples = []
    running_features = []

    for file in tqdm(os.listdir(examples_dir)):
        # Load, extract feature information and
        # examples from file
        path = os.path.join(examples_dir, file)
        with open(path, "r") as f:
            examples = json.load(f)
        module, feature = to_feature(file)

        # Can only score one module at a time
        if counter == 0:
            current_module = module

        # Iterate until reach batch size
        if counter < batch_size or current_module == module:
            running_examples.append(examples)
            running_features.append(feature)
            counter += 1
            continue

        # Score batch
        scores = _score(
            model, submodule_dict[module], running_examples, feature, generation_size
        )

        # Save scores
        save(examples, scores, path)

        # Reset
        running_examples = []
        counter = 0


def to_feature(string):
    string = string.replace(".txt", "").replace("feature", "")
    module, feature = string.split("_")
    return module, int(feature)


def _score(
    model,
    submodule: Dict,
    examples: List[List[str]],
    features: List[int],
    generation_size: int,
):
    flattened_examples = sum(examples, [])

    indices = torch.arange(len(flattened_examples))
    splits = indices.split(generation_size)

    all_scores = []

    with model.trace(flattened_examples):
        scores = submodule.ae.output

        for feature, split in zip(features, splits):
            score = torch.any(scores[split, :, feature] != 0, dim=0)

            score = score.sum().item()

            all_scores.append(scores.save())

    return map(lambda x: x.value, all_scores)


def save(examples: List, scores: List[int], path: str):
    for examples, score in zip(examples, scores):
        result = {"examples": examples, "score": score}

        with open(path, "wb") as f:
            f.write(orjson.dumps(result))
>>>>>>> d3b81f50d676295bd88d0e499194b44c40c02b13
