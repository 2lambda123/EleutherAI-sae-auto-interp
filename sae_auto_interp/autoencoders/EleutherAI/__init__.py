from .model import Sae
from typing import List
from ..latents_wrapper import AutoencoderLatents
from functools import partial
from ..OpenAI.model import ACTIVATIONS_CLASSES, TopK
    
DEVICE = "cuda:0"

def load_eai_autoencoders(
    model, 
    ae_layers: List[int], 
    weight_dir:str
):
    submodules = {}

    for layer in ae_layers:
        if "llama" in weight_dir:
            model_type = "llama"
            path = f"{weight_dir}/layers.{layer}"
        
        if "gpt2" in weight_dir:
            model_type = "gpt2"
            path = f"{weight_dir}/h.{layer}"
        
        sae = Sae.load_from_disk(path, DEVICE)

        def _forward(sae, x):
            latents = sae.pre_acts(x)
            trained_k = sae.cfg.k
            topk = TopK(trained_k, postact_fn=ACTIVATIONS_CLASSES["Identity"]())
            return topk(latents)
        if model_type == "llama":
            submodule = model.model.layers[layer]
        if model_type == "gpt2":
            submodule = model.transformer.h[layer]
        if sae.cfg.num_latents!=0:
            num_latents = sae.cfg.num_latents
        else:
            num_latents = sae.d_in * sae.cfg.expansion_factor
        submodule.ae = AutoencoderLatents(sae,partial(_forward, sae),num_latents)

        submodules[submodule._module_path] = submodule
    
    with model.edit(" "):
        for _, submodule in submodules.items():
            acts = submodule.output[0]
            submodule.ae(acts, hook=True)

    return submodules	