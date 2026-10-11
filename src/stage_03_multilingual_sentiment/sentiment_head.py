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
    p_negative = probabilities[:, 0]
    p_positive = probabilities[:, 2]
    sentiment_score = p_positive - p_negative
    return sentiment_score

def compute_controversy_score(probabilities: np.ndarray) -> np.ndarray:
    """
    Controversy Score (Polarization & Heated Debate Index):
    Measures the intensity of debate between opposing fanbases (e.g. referee scandals, player drama).
    High controversy occurs when both Positive and Negative probabilities are elevated and Neutral is low.
    Bounded in [0.0, 1.0].
    """
    p_neg = probabilities[:, 0]
    p_pos = probabilities[:, 2]
    p_neu = probabilities[:, 1]
    
    # Polarized contention between opposing sentiment poles
    polarization = 2.0 * np.minimum(p_pos, p_neg) / (p_pos + p_neg + 1e-6)
    # Weigh by emotional involvement (non-neutral weight)
    emotional_weight = (1.0 - p_neu).clip(0.0, 1.0)
    controversy = polarization * emotional_weight
    return np.clip(controversy, 0.0, 1.0)
