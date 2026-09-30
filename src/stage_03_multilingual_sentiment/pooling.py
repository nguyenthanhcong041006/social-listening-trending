import torch
import numpy as np

def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
    """
    Mean Pooling:
    Computes average vector representation across token embeddings weighted by attention_mask (ignoring padding tokens).
    Yields dense contextual sentence embeddings for social posts.
    
    Formula:
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        sentence_embeddings = sum_embeddings / sum_mask
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    sentence_embeddings = sum_embeddings / sum_mask
    return sentence_embeddings.cpu().numpy()
