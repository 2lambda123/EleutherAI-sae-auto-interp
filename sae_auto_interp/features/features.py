from dataclasses import dataclass
<<<<<<< HEAD
from typing import List, Tuple, Callable, Optional
import torch
from tqdm import tqdm
import orjson
import blobfile as bf
from dataclasses import asdict

from torch import Tensor
from torchtyping import TensorType

from ..logger import logger
from .sampling import default_sampler
from .example import load_examples
from .activations import pool_max_activation_slices, get_random_tokens

from .config import FeatureConfig
=======

import blobfile as bf
import orjson
from torchtyping import TensorType


@dataclass
class Example:
    tokens: TensorType["seq"]
    activations: TensorType["seq"]

    def __hash__(self) -> int:
        return hash(tuple(self.tokens.tolist()))

    def __eq__(self, other: "Example") -> bool:
        return self.tokens.tolist() == other.tokens.tolist()

    @property
    def max_activation(self):
        return max(self.activations)

    @staticmethod
    def prepare_examples(tokens, activations):
        return [
            Example(
                tokens=toks,
                activations=acts,
            )
            for toks, acts in zip(tokens, activations)
        ]

>>>>>>> d3b81f50d676295bd88d0e499194b44c40c02b13

@dataclass
class FeatureID:
    module_name: int
    feature_index: int

    def __repr__(self) -> str:
        return f"{self.module_name}_feature{self.feature_index}"
<<<<<<< HEAD
    
    @staticmethod
    def from_path(path: str) -> 'FeatureID':
        path = path.replace(".txt", "").replace("feature", "") 
        module_name, feature_index = path.split("_")
        return FeatureID(module_name, int(feature_index))
    

class FeatureRecord:
    def __init__(self, feature: FeatureID):
=======


class FeatureRecord:
    def __init__(
        self,
        feature: Feature,
    ):
>>>>>>> d3b81f50d676295bd88d0e499194b44c40c02b13
        self.feature = feature

        self.examples = []
        self.train = []
        self.test = []
        self.random_examples = []

    @property
<<<<<<< HEAD
    def max_activation(self) -> float:
        return self.examples[0].max_activation if self.examples else 0.0
    
    @classmethod
    def load(
        self,
        feature: FeatureID,
        cfg: FeatureConfig,
        tokens: TensorType["batch", "seq"],
        feature_locations: TensorType["n_locations", 2],
        feature_activations: TensorType["n_locations"],
        sampler: Callable,
        processed_dir: Optional[str],
        n_random_examples: int,
    ) -> None:
        record = FeatureRecord(feature)

        processed_tokens, processed_activations = pool_max_activation_slices(
            feature_locations, 
            feature_activations, 
            tokens, 
            ctx_len=20, 
            k=cfg.max_examples
        )

        if len(processed_tokens) < cfg.min_examples:
            logger.error(f"Not enough examples: {feature}.")
            raise ValueError(f"Not enough examples: {feature}.")

        record.examples = load_examples(processed_tokens, processed_activations)

        if cfg.sample:
            sampler(record, **asdict(cfg))

        if processed_dir:
            record._load_processed(processed_dir)
        
        if n_random_examples > 0:
            record._load_random_examples(
                n_random_examples, tokens, feature_locations
            )

        return record
=======
    def max_activation(self):
        return self.examples[0].max_activation
>>>>>>> d3b81f50d676295bd88d0e499194b44c40c02b13

    def _load_random_examples(
        self, 
        n_random_examples: int,
        tokens: TensorType["batch", "seq"],
        feature_locations: TensorType["batch", 2],
    ) -> None:
        random_tokens = get_random_tokens(
            feature_locations, tokens, n_random_examples
        )

        self.random_examples = load_examples(
            random_tokens, torch.zeros_like(random_tokens)
        )

    def _load_processed(self, processed_dir: str) -> None:
        path = f"{processed_dir}/{self.feature}.json"
        with bf.BlobFile(path, "rb") as f:
            processed_data = orjson.loads(f.read())
            self.__dict__.update(processed_data)
<<<<<<< HEAD
    
    def save(self, directory: str, save_examples: bool = False) -> None:
=======

    def save(self, directory: str, save_examples=False):
>>>>>>> d3b81f50d676295bd88d0e499194b44c40c02b13
        path = f"{directory}/{self.feature}.json"
        serializable = self.__dict__.copy()

        if not save_examples:
            serializable.pop("examples", None)
            serializable.pop("train", None)
            serializable.pop("test", None)

        serializable.pop("feature", None)
        with bf.BlobFile(path, "wb") as f:
            f.write(orjson.dumps(serializable))
<<<<<<< HEAD


def _from_tensor(
    raw_dir: str,
    module_name: str,
    selected_features: Optional[List[int]] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    locations_path = f"{raw_dir}/{module_name}_locations.pt"
    activations_path = f"{raw_dir}/{module_name}_activations.pt"
    
    locations = torch.load(locations_path)
    activations = torch.load(activations_path)

    features = torch.unique(locations[:, 2])

    if selected_features is not None:
        selected_features_tensor = torch.tensor(selected_features)
        features = features[torch.isin(features, selected_features_tensor)]

    return locations, activations, features


def load_feature_batch(
    cfg: FeatureConfig,
    tokens: TensorType["batch", "seq"],
    module_name: str,
    raw_dir: str,
    processed_dir: Optional[str] = None,
    n_random_examples: int = 10,
    selected_features: Optional[List[int]] = None,
    sampler: Callable = default_sampler,
) -> List[FeatureRecord]:
    
    locations, activations, features = \
        _from_tensor(
            raw_dir, module_name, selected_features
        )
    
    records = []

    for feature_index in tqdm(features, desc=f"Loading {module_name}"):
        mask = locations[:, 2] == feature_index
        feature_locations = locations[mask][:,:2]
        feature_activations = activations[mask]
    
        feature = FeatureID(module_name, feature_index.item())

        try:
            record = FeatureRecord.load(
                feature,
                tokens=tokens,
                cfg=cfg,
                feature_locations=feature_locations,
                feature_activations=feature_activations,
                processed_dir=processed_dir,
                n_random_examples=n_random_examples,
                sampler=sampler,
            )

            records.append(record)

        except ValueError as e:
            logger.error(f"Error loading feature {feature}: {e}")
            continue

    return records
=======
>>>>>>> d3b81f50d676295bd88d0e499194b44c40c02b13
