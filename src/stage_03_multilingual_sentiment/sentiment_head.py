import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict

class SentimentClassificationHead:
    """
    Sentiment Classification Head:
    Takes logits from XLM-RoBERTa and applies Softmax to compute probabilities across 3 classes:
    - 0: Negative
    - 1: Neutral
    - 2: Positive
    """

    def __init__(self, id2label: Dict[int, str] = None):
        self.id2label = id2label or {0: "negative", 1: "neutral", 2: "positive"}

    def predict_probabilities(self, logits: torch.Tensor) -> np.ndarray:
        """Computes class probability distribution via Softmax."""
        probs = F.softmax(logits, dim=-1)
        return probs.cpu().numpy()

def compute_sentiment_score(probabilities: np.ndarray) -> np.ndarray:
    """
    Sentiment Score:
    Applies the mathematical formulation from the pipeline architecture:
        S = P(positive) - P(negative)
    Where S is bounded in [-1.0, 1.0]:
    - S > 0: Leaning Positive
    - S = 0: Neutral
    - S < 0: Leaning Negative
    """
    # probabilities: shape (N, 3) where columns correspond to [P_neg, P_neu, P_pos]
    p_negative = probabilities[:, 0]
    p_positive = probabilities[:, 2]
    sentiment_score = p_positive - p_negative
    return sentiment_score
