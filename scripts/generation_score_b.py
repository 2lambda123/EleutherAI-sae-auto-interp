# %%
import asyncio
import random
from nnsight import LanguageModel
import os

from sae_auto_interp.scorers.generation.utils import score
from sae_auto_interp.utils import load_tokenized_data
from sae_auto_interp.autoencoders import load_autoencoders

model, submodule_dict = load_autoencoders(
    "openai-community/gpt2",
    list(range(0, 4, 2)),
    "/share/u/caden/sae-auto-interp/sae_auto_interp/autoencoders/OpenAI/gpt2_128k",
)
tokens = load_tokenized_data(model.tokenizer)

raw_features_path = "raw_features"
explanations_dir = "results/explanations/simple"
scorer_out_dir = "results/scores"
random.seed(22)

scorer_inputs = []

# %%
score(
    model,
    submodule_dict,
    examples_dir="/share/u/caden/sae-auto-interp/results/scores/generation_a",
    batch_size=10
)

score

# %%

# c=  ['affewfw awdwaewf']

model.tokenizer(["adawdawd", "adawdawdawdawdw"], return_tensors='pt', padding=True, truncation=True)['input_ids'].shape