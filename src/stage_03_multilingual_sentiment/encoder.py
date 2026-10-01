import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from typing import Dict, Any, List, Optional
from loguru import logger
from .pooling import mean_pooling

class XLMRoBERTaEmbeddingEncoder:
    """
    XLM-RoBERTa Embedding Encoder:
    Dedicated Transformer backbone fine-tuned for semantic textual representations and topic modeling.
    Extracts dense contextual token representations (last_hidden_state) for attention-masked Mean Pooling.
    Used downstream by Stage 4 (BERTopic) for semantic topic clustering.
    """

    def __init__(self, model_name: str = "xlm-roberta-base", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        logger.info(f"Initializing XLM-RoBERTa Embedding Model ({model_name}) on device: {self.device}")
        
        try:
            # Try loading directly from local cache or local path (instant & 100% offline-ready)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_name, local_files_only=True).to(self.device)
        except Exception:
            # Fall back to downloading from Hugging Face Hub if not cached yet
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def tokenize(self, texts: List[str], max_length: int = 128) -> Dict[str, torch.Tensor]:
        """Tokenizes text list and transfers tensors to target compute device."""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)

    def forward(self, encoded_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Embedding Model.
        Returns last_hidden_state and attention_mask.
        """
        with torch.no_grad():
            outputs = self.model(**encoded_inputs)
        return {
            "hidden_states": outputs.last_hidden_state,
            "attention_mask": encoded_inputs["attention_mask"]
        }

    def encode(self, texts: List[str], batch_size: int = 32, max_length: int = 128, normalize: bool = True):
        """Encodes texts into pooled L2-normalized sentence embeddings."""
        import numpy as np
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = self.tokenize(batch_texts, max_length=max_length)
            outputs = self.forward(encoded)
            embeds = mean_pooling(outputs["hidden_states"], outputs["attention_mask"], normalize=normalize)
            all_embeddings.append(embeds)
        return np.vstack(all_embeddings) if all_embeddings else np.empty((0, self.model.config.hidden_size))


class XLMRoBERTaSentimentClassifier:
    """
    XLM-RoBERTa Sentiment Classifier:
    Dedicated Transformer fine-tuned for multilingual sequence classification (Positive/Negative/Neutral).
    Outputs raw logits from the sequence classification head.
    """

    def __init__(self, model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        logger.info(f"Initializing XLM-RoBERTa Sentiment Model ({model_name}) on device: {self.device}")
        
        try:
            # Try loading directly from local cache or local path (instant & 100% offline-ready)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True).to(self.device)
        except Exception:
            # Fall back to downloading from Hugging Face Hub if not cached yet
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def tokenize(self, texts: List[str], max_length: int = 128) -> Dict[str, torch.Tensor]:
        """Tokenizes text list and transfers tensors to target compute device."""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)

    def forward(self, encoded_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through Sentiment Model.
        Returns classification logits.
        """
        with torch.no_grad():
            outputs = self.model(**encoded_inputs)
        return outputs.logits


class XLMRoBERTaDualEncoder:
    """
    Unified manager for dual fine-tuned XLM-RoBERTa models:
    1. Embedding Model: fine-tuned for sentence semantic similarity & BERTopic clustering.
    2. Sentiment Model: fine-tuned for multilingual sentiment polarity classification.
    
    Provides clean decoupled inference and backward-compatible interfaces.
    """

    def __init__(
        self,
        embedding_model_name: Optional[str] = None,
        sentiment_model_name: Optional[str] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Resolve model identifiers
        self.embedding_model_name = embedding_model_name or model_name or "xlm-roberta-base"
        self.sentiment_model_name = sentiment_model_name or model_name or "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        
        # If both point to the exact same sentiment classification checkpoint (shared fallback mode)
        self.is_shared = (self.embedding_model_name == self.sentiment_model_name)
        
        if self.is_shared:
            logger.info(f"Using single shared XLM-RoBERTa backbone ({self.embedding_model_name}) for dual outputs.")
            self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.embedding_model_name,
                output_hidden_states=True
            ).to(self.device)
            self.model.eval()
            self.embedding_encoder = None
            self.sentiment_classifier = None
        else:
            logger.info("Initializing 2 dedicated fine-tuned XLM-RoBERTa models:")
            logger.info(f" - Model 1 (BERTopic Embeddings): {self.embedding_model_name}")
            logger.info(f" - Model 2 (Sentiment Analysis):   {self.sentiment_model_name}")
            self.embedding_encoder = XLMRoBERTaEmbeddingEncoder(model_name=self.embedding_model_name, device=self.device)
            self.sentiment_classifier = XLMRoBERTaSentimentClassifier(model_name=self.sentiment_model_name, device=self.device)
            self.tokenizer = self.embedding_encoder.tokenizer
            self.model = getattr(self.sentiment_classifier, "model", None)

    def tokenize(self, texts: List[str], max_length: int = 128) -> Dict[str, torch.Tensor]:
        """Tokenize using primary tokenizer (or embedding tokenizer)."""
        if self.is_shared:
            return self.tokenizer(
                texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            ).to(self.device)
        return self.embedding_encoder.tokenize(texts, max_length=max_length)

    def forward(self, encoded_inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Legacy forward interface for shared model mode."""
        if self.is_shared:
            with torch.no_grad():
                outputs = self.model(**encoded_inputs)
            return {
                "logits": outputs.logits,
                "hidden_states": outputs.hidden_states[-1],
                "attention_mask": encoded_inputs["attention_mask"]
            }
        raise NotImplementedError(
            "Direct forward(encoded_inputs) is only supported in shared model mode. "
            "For dual decoupled models, use forward_embedding() and forward_sentiment()."
        )

    def forward_embedding(self, texts: List[str], max_length: int = 128) -> Dict[str, torch.Tensor]:
        """Tokenizes and forwards text batch through embedding encoder."""
        if self.is_shared:
            encoded = self.tokenize(texts, max_length=max_length)
            with torch.no_grad():
                outputs = self.model(**encoded)
            return {
                "hidden_states": outputs.hidden_states[-1],
                "attention_mask": encoded["attention_mask"]
            }
        encoded = self.embedding_encoder.tokenize(texts, max_length=max_length)
        return self.embedding_encoder.forward(encoded)

    def forward_sentiment(self, texts: List[str], max_length: int = 128) -> torch.Tensor:
        """Tokenizes and forwards text batch through sentiment classifier, returning logits."""
        if self.is_shared:
            encoded = self.tokenize(texts, max_length=max_length)
            with torch.no_grad():
                outputs = self.model(**encoded)
            return outputs.logits
        encoded = self.sentiment_classifier.tokenize(texts, max_length=max_length)
        return self.sentiment_classifier.forward(encoded)


# Alias for backward compatibility
XLMRoBERTaEncoder = XLMRoBERTaDualEncoder
