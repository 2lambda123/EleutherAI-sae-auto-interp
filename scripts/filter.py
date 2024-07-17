# %%

from tqdm import tqdm
from sae_auto_interp.utils import load_tokenized_data
from sae_auto_interp.autoencoders import load_autoencoders
from sae_auto_interp.features import FeatureRecord
import torch
from sae_auto_interp.filters.stats import Unigram, CombinedStat
from collections import defaultdict

model, submodule_dict = load_autoencoders(
    "openai-community/gpt2", 
    list(range(0,12,2)),
    "/share/u/caden/sae-auto-interp/sae_auto_interp/autoencoders/OpenAI/gpt2"
)

tokens = load_tokenized_data(model.tokenizer)


raw_features_path = "/share/u/caden/sae-auto-interp/raw_features"
processed_features_path = "/share/u/caden/sae-auto-interp/feature_statistics"

k_values = torch.arange(10, 100, 10)

results = defaultdict(lambda: defaultdict(list))

for layer, submodule in submodule_dict.items():
    ae = submodule.ae._module

    records = FeatureRecord.from_tensor(
        tokens,
        module_name=layer,
        tokenizer=model.tokenizer,
        raw_dir=raw_features_path,
        selected_features=list(range(10)),
        min_examples=50,
        max_examples=5000,
    )

    activation = Unigram(
        k=20,
    )
    
    for k in tqdm(k_values):
        activation.refresh(k=k)
        activation.compute(records)
        
        for record in records:
            if len(record.examples) < k:
                continue

            n_unique = record.unique_tokens
            results[layer][k.item()].append(n_unique)


    break


