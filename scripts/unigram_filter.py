# %%

import torch
from tqdm import tqdm
from collections import defaultdict
from nnsight import LanguageModel

from sae_auto_interp.utils import load_tokenized_data
from sae_auto_interp.features import FeatureRecord
from sae_auto_interp.filters import Unigram

model = LanguageModel("openai-community/gpt2")
tokens = load_tokenized_data(model.tokenizer)

raw_features_path = "/share/u/caden/sae-auto-interp/raw_features"
processed_features_path = "/share/u/caden/sae-auto-interp/feature_statistics"

k_values = torch.arange(50, 1550, 150)

results = defaultdict(lambda: defaultdict(list))

# %%

sparse = {}

for i in range(0,12,2):
    module_name = f".transformer.h.{i}"

    records = FeatureRecord.from_tensor(
        tokens,
        module_name=module_name,
        raw_dir=raw_features_path,
        selected_features=list(range(100)),
        min_examples=1,
        max_examples=5000,
    )

    activation = Unigram(
        k=20,
    )
    
    # for k in tqdm(k_values):
    #     activation.refresh(k=k)
    #     activation.compute(records)

    #     for record in records:
    #         if len(record.examples) < k:
    #             continue
        
    #         results[i][k.item()].append(record.n_unique)

    # check at medium k and save
    activation.refresh(k=1000)
    activation.compute(records)

    for record in records:
        if len(record.examples) < 1000:
            continue
    
        sparse[str(record.feature)] = record.n_unique


# %%
import matplotlib.pyplot as plt
import collections

# Group y_values by their corresponding k values
data = collections.defaultdict(list)

for i in results:
    for k, records in results[i].items():
        for record in records:
            data[k].append(record)

# Prepare data for the box plot
x_labels = sorted(data.keys())
box_data = [data[k] for k in x_labels]

# Create the box plot
plt.figure(figsize=(10, 6))
plt.boxplot(box_data, patch_artist=True)

# Set the x-tick labels to the unique k values
plt.xticks(range(1, len(x_labels) + 1), x_labels)

plt.xlabel('k')
plt.ylabel('n_unique')
plt.title('Box Plot of n_unique for increasing k')

plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()

# %%

plt.hist(sparse.values(), bins=100)
plt.xlabel('n_unique')
plt.ylabel('Frequency')
plt.title('Histogram of n_unique for k=1000')