<<<<<<< HEAD
from .model import Sae
from typing import List
from ..latents_wrapper import AutoencoderLatents
=======
>>>>>>> d3b81f50d676295bd88d0e499194b44c40c02b13
from functools import partial
from typing import List

from ..OpenAI.model import ACTIVATIONS_CLASSES, TopK
from ..wrapper import AutoencoderLatents
from .model import Sae

DEVICE = "cuda:0"


def load_eai_autoencoders(model, ae_layers: List[int], weight_dir: str):
    submodules = {}

    for layer in ae_layers:
        path = f"{weight_dir}/layers.{layer}"
<<<<<<< HEAD
        if "llama" in weight_dir:
            model_type = "llama"
            path = f"{weight_dir}/layers.{layer}"
        
        if "gpt2" in weight_dir:
            model_type = "gpt2"
            path = f"{weight_dir}/h.{layer}"
        
=======
>>>>>>> d3b81f50d676295bd88d0e499194b44c40c02b13
        sae = Sae.load_from_disk(path, DEVICE)

        def _forward(sae, x):
            latents = sae.pre_acts(x)
            trained_k = sae.cfg.k
            topk = TopK(trained_k, postact_fn=ACTIVATIONS_CLASSES["Identity"]())
<<<<<<< HEAD
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

        submodule = model.model.layers[layer]
        submodule.ae = AutoencoderLatents(sae,partial(_forward, sae),sae.d_in * sae.cfg.expansion_factor)

        submodules[submodule._module_path] = submodule
=======
            return topk(encoded)
        if "llama" in weight_dir:
            submodule = model.model.layers[layer]
        elif "gpt2" in weight_dir:
            submodule = model.transformer.h[layer]

        submodule.ae = AutoencoderLatents(
            sae, partial(_forward, sae), width=sae.d_in * sae.cfg.expansion_factor
        )

        submodules[layer] = submodule
>>>>>>> d3b81f50d676295bd88d0e499194b44c40c02b13
    
    with model.edit(" "):
        for _, submodule in submodules.items():
            acts = submodule.output[0]
            submodule.ae(acts, hook=True)

<<<<<<< HEAD
    return submodules ,model	
=======
    return submodules
>>>>>>> d3b81f50d676295bd88d0e499194b44c40c02b13
