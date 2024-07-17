import json
import torch

from sae_auto_interp.autoencoders import load_autoencoders
from sae_auto_interp.features import FeatureCache
from sae_auto_interp.utils import load_tokenized_data

model, submodule_dict = load_autoencoders(
    "openai-community/gpt2", 
    list(range(0,12,2)),
    "/share/u/caden/sae-auto-interp/sae_auto_interp/autoencoders/OpenAI/gpt2_128k",
)
tokens = load_tokenized_data(model.tokenizer)

names = [
    model.transformer.h[i]._module_path
    for i in range(0,12,2)
]

with open("/share/u/caden/sae-auto-interp/scripts/unique.json") as f:
    data = json.load(f)

module_filter = { 
    name : torch.tensor(data[name], device="cuda:0") 
    for name in names
}

cache = FeatureCache(
    model, 
    submodule_dict, 
    filters=module_filter
)

cache.run(tokens, n_tokens=15_000_000)

cache.save(save_dir="/share/u/caden/sae-auto-interp/raw_features")