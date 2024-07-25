# %% 

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

from sae_auto_interp.explainers.simple.prompt_builder import build_prompt


prompt = build_prompt(
    "", False, False
)

len(tokenizer.apply_chat_template(
    prompt,
    # tokenize=False,
    add_generation_prompt=True
))

# %%

778.13 - 725