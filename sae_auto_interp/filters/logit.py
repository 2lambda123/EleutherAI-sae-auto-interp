import torch
from .stats import Stat

class Logits(Stat):
    collated = True

    def __init__(self, 
        tokenizer, 
        k=10,
        W_U = None,
    ):
        self.tokenizer = tokenizer
        self.k = k
        self.W_U = W_U

    def refresh(self, W_dec=None, **kwargs):
        self.W_dec = W_dec
        
    def compute(self, records, **kwargs):

        feature_indices = [
            record.feature.feature_index 
            for record in records
        ]
        
        narrowed_logits = torch.matmul(
            self.W_U, 
            self.W_dec[:,feature_indices]
        )

        top_logits = torch.topk(
            narrowed_logits, self.k, dim=0
        ).indices

        per_example_top_logits = top_logits.T

        for record_index, record in enumerate(records):

            record.top_logits = \
                self.tokenizer.batch_decode(
                    per_example_top_logits[record_index]
                )
