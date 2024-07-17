# %%
import torch
from sae_auto_interp.autoencoders import load_autoencoders
from sae_auto_interp.features import FeatureCache
import json

model, submodule_dict = load_autoencoders(
    "openai-community/gpt2", 
    list(range(0,12,2)),
    "/share/u/caden/sae-auto-interp/sae_auto_interp/autoencoders/OpenAI/gpt2_128k",
)

# %%

from sae_auto_interp.filters.neighbor import get_neighbors

feature_filter = {
    module_path : list(range(100))
    for module_path in submodule_dict.keys()
}

neighbors, unique = get_neighbors(submodule_dict, feature_filter=feature_filter)

# %%

with open("unique.json", "w") as f:
    json.dump(unique, f, indent=4)