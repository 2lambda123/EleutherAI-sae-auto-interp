from .EleutherAI import load_eai_autoencoders
from .OpenAI import load_oai_autoencoders
from .Sam import load_sam_autoencoders
from nnsight import LanguageModel

def load_model(model_name, device_map="auto", dispatch="true", **kwargs):
    model = LanguageModel(
        model_name, 
        device_map=device_map,
        dispatch=dispatch,
        **kwargs
    )
    return model

def load_autoencoders(model_name, ae_layers, weight_dir, **kwargs):

    model = load_model(model_name, **kwargs)

    if "gpt2_128k" in weight_dir or "gpt2_32k" in weight_dir:
        submodules = load_oai_autoencoders(model, ae_layers, weight_dir)
       
    if "llama" in weight_dir or "nora" in weight_dir:
        submodules = load_eai_autoencoders(model, ae_layers, weight_dir)

    if "pythia" in weight_dir:
        submodules = load_sam_autoencoders(model, ae_layers, weight_dir, **kwargs)
    
    return model, submodules