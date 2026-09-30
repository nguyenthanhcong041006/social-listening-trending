import torch
import numpy as np

def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 Normalization: embedding = L2_normalize(embedding) as specified in PDF Section 5.1."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, eps, norms)
    return vectors / norms

def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor, normalize: bool = True) -> np.ndarray:
    """
    Mean Pooling:
    Computes average vector representation across token embeddings weighted by attention_mask (ignoring padding tokens).
    Yields dense contextual sentence embeddings for social posts.
    
    Formula (PDF Section 5.1):
        h_1, h_2, ..., h_n = XLM-RoBERTa(text)
        embedding = Σ(h_i × mask_i) / Σ(mask_i)
        embedding = L2_normalize(embedding)
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    sentence_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
    
    if normalize:
        sentence_embeddings = l2_normalize(sentence_embeddings)
        
    return sentence_embeddings
