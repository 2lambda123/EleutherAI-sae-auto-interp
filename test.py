# %%
import random
from nnsight import LanguageModel
from sae_auto_interp.utils import load_tokenized_data
from sae_auto_interp.features import FeatureRecord, load_feature_batch, FeatureConfig

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
tokens = load_tokenized_data(model.tokenizer)

raw_features_path = "raw_features"
explainer_out_dir = "results/explanations/simple"

explainer_inputs=[]
random.seed(22)

for layer in range(0,12,2):
    module_name = f".transformer.h.{layer}"

    records = load_feature_batch(
        FeatureConfig(),
        tokens,
        module_name,
        selected_features=list(range(20)),
        raw_dir = raw_features_path,
    )

    break


# %%

directory = "tests"
path = f"{directory}/{records[0].feature}.json"
records[0].examples = records[0].examples[:100]

for example in records[0].examples:
    example.activations = example.activations.tolist()
    example.tokens = example.tokens.tolist()

serializable = records[0].__dict__.copy()

import orjson
import blobfile as bf

serializable.pop("feature", None)
serializable.pop('random_examples', None)
serializable.pop('random_example', None)
serializable.pop('random_example_idx', None)
with bf.BlobFile(path, "wb") as f:
    f.write(orjson.dumps(serializable))

# %%
serializable