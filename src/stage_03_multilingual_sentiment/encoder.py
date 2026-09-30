import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any
from loguru import logger

class XLMRoBERTaEncoder:
    """
    XLM-RoBERTa Encoder:
    Multilingual Transformer Foundation Model.
    Supports concurrent dual-branch processing:
    1. Extracts last_hidden_state for Mean Pooling (Contextual Sentence Embeddings).
    2. Passes outputs through Classification Head to predict sentiment probabilities (Positive/Negative/Neutral).
    """

    def __init__(self, model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing XLM-RoBERTa ({model_name}) on device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            output_hidden_states=True
        ).to(self.device)
        self.model.eval()

    def tokenize(self, texts: list, max_length: int = 128) -> Dict[str, torch.Tensor]:
        """Tokenizes text list and transfers tensors to target compute device."""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)

    def forward(self, encoded_inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Forward pass through XLM-RoBERTa.
        Returns last_hidden_state and classification logits.
        """
        with torch.no_grad():
            outputs = self.model(**encoded_inputs)
        return {
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states[-1], # last hidden state
            "attention_mask": encoded_inputs["attention_mask"]
        }
