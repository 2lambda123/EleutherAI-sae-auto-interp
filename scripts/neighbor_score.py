# %%
import asyncio
import random
import json
from nnsight import LanguageModel

from sae_auto_interp.scorers import NeighborScorer, ScorerInput
from sae_auto_interp.clients import get_client, execute_model
from sae_auto_interp.utils import load_tokenized_data, load_explanation
from sae_auto_interp.features import FeatureRecord
from sae_auto_interp.scorers.neighbor.utils import load_neighbors 

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
tokens = load_tokenized_data(model.tokenizer)

raw_features_path = "raw_features"
explanations_dir = "results/explanations/simple"
scorer_out_dir = "results/scores"
random.seed(22)

with open("scripts/unique.json", "r") as f:
    unique = json.load(f)


scorer_inputs = []

for layer in range(0,12,2):
    module_name = f".transformer.h.{layer}"

    all_records = FeatureRecord.from_tensor(
        tokens,
        module_name,
        selected_features=unique[module_name],
        raw_dir = raw_features_path,
        min_examples=120,
        max_examples=10000
    )

    records = all_records[:10]
    load_neighbors(records, all_records, module_name, "scripts/neighbors.json")

    for record in records:

        try:
            examples = record.examples
            test_examples = random.sample(examples[100:200], 20)
            explanation = load_explanation(record.feature, explanations_dir)
            scorer_inputs.append(
                ScorerInput(
                    explanation=explanation,
                    record=record,
                    test_examples=test_examples,
                )
            )
        except Exception as e:
            print(e)
            continue

    break

client = get_client("outlines", "astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit")

scorer = NeighborScorer(
    client,
    model.tokenizer
)

asyncio.run(
    execute_model(
        scorer, 
        scorer_inputs,
        output_dir=scorer_out_dir,
        record_time=True
    )
)

