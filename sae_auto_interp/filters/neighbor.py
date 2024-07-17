import torch
import torch.nn.functional as F
from collections import defaultdict
from .stats import Stat
import umap 
from sklearn.neighbors import NearestNeighbors

def cos(matrix, selected_features=[0]):
    
    a = matrix[:,selected_features]
    b = matrix   

    a = F.normalize(a, p=2, dim=0)
    b = F.normalize(b, p=2, dim=0)

    cos_sim = torch.mm(a.t(), b)

    return cos_sim

def get_neighbors(submodule_dict, feature_filter, k=10):
    """
    Get the required features for neighbor scoring.

    Returns:
        neighbors_dict: Nested dictionary of modules -> neighbors -> indices, values
        per_layer_features (dict): A dictionary of features per layer
    """

    neighbors_dict = defaultdict(dict)
    per_layer_features = {}
    
    for module_path, submodule in submodule_dict.items():
        selected_features = feature_filter.get(module_path, False)
        if not selected_features:
            continue

        W_D = submodule.ae.autoencoder._module.decoder.weight
        cos_sim = cos(W_D, selected_features=selected_features)
        top = torch.topk(cos_sim, k=k)

        top_indices = top.indices   
        top_values = top.values

        for i, (indices, values) in enumerate(zip(top_indices, top_values)):
            neighbors_dict[module_path][i] = {
                "indices": indices.tolist()[1:],
                "values": values.tolist()[1:]
            }
        
        per_layer_features[module_path] = torch.unique(top_indices).tolist()

    return neighbors_dict, per_layer_features


class UmapNeighbors(Stat):

    def __init__(
        self, 
        n_neighbors: int = 15, 
        metric: str = 'cosine', 
        min_dist: float = 0.05, 
        n_components: int = 2, 
        random_state: int = 42,
        **kwargs
    ):
        self.umap_model = umap.UMAP(
            n_neighbors=n_neighbors, 
            metric=metric, 
            min_dist=min_dist, 
            n_components=n_components, 
            random_state=random_state,
            **kwargs
        )

    def refresh(self, W_dec=None, **kwargs):
        self.embedding = \
            self.umap_model.fit_transform(W_dec)

    def compute(self, records, *args, **kwargs): 
        for record in records:
            self._compute(record, *args, **kwargs)

    def _compute(self, record, *args, **kwargs):
        # Increment n_neighbors to account for query
        n_neighbors = n_neighbors + 1
        feature_index = record.feature.feature_index
        query = self.embedding[feature_index]

        nn_model = NearestNeighbors(n_neighbors=n_neighbors)
        nn_model.fit(self.embedding)

        distances, indices = nn_model.kneighbors([query])

        neighbors = {
            'distances': distances[0,1:].tolist(),
            'indices': indices[0,1:].tolist()
        }

        record.neighbors = neighbors

class CosNeigbors(Stat):
    pass