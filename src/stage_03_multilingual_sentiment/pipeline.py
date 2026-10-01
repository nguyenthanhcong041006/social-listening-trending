import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from loguru import logger
from .encoder import XLMRoBERTaDualEncoder, XLMRoBERTaEncoder
from .pooling import mean_pooling
from .sentiment_head import SentimentClassificationHead, compute_sentiment_score
from src.utils.file_io import save_dataframe, load_dataframe, ensure_dir

class MultilingualSentimentPipeline:
    """
    Stage 3 Pipeline:
    Executes dual-branch processing using two dedicated fine-tuned XLM-RoBERTa models:
    1. Branch A: Contextual Sentence Embeddings (fine-tuned for BERTopic topic clustering)
    2. Branch B: Multilingual Sentiment Analysis (fine-tuned for 3-class sentiment score)
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        
        # Support dual fine-tuned models with backward-compatible fallbacks
        self.embedding_model_name = self.config.get(
            "embedding_model_name", 
            self.config.get("model_name", "xlm-roberta-base")
        )
        self.sentiment_model_name = self.config.get(
            "sentiment_model_name", 
            self.config.get("model_name", "cardiffnlp/twitter-xlm-roberta-base-sentiment")
        )
        self.batch_size = self.config.get("batch_size", 32)
        self.max_length = self.config.get("max_length", 128)
        
        self.dual_encoder = XLMRoBERTaDualEncoder(
            embedding_model_name=self.embedding_model_name,
            sentiment_model_name=self.sentiment_model_name
        )
        self.encoder = self.dual_encoder  # Backward compatibility alias
        self.head = SentimentClassificationHead()
        
        self.output_df_path = self.config.get(
            "output_df_path", "data/03_embeddings_sentiment/posts_with_sentiment.parquet"
        )
        self.output_embeddings_path = self.config.get(
            "output_embeddings_path", "data/03_embeddings_sentiment/sentence_embeddings.npy"
        )

    def extract_embeddings(self, texts: list) -> np.ndarray:
        """Extracts contextual sentence embeddings using the fine-tuned embedding model."""
        total_samples = len(texts)
        logger.info(f"[Branch A] Extracting BERTopic Sentence Embeddings ({self.embedding_model_name}) for {total_samples} posts...")
        
        all_embeddings = []
        for i in tqdm(range(0, total_samples, self.batch_size), desc="Embedding Inference"):
            batch_texts = texts[i : i + self.batch_size]
            outputs = self.dual_encoder.forward_embedding(batch_texts, max_length=self.max_length)
            batch_embeds = mean_pooling(outputs["hidden_states"], outputs["attention_mask"], normalize=True)
            all_embeddings.append(batch_embeds)
            
        return np.vstack(all_embeddings) if all_embeddings else np.empty((0, 768))

    def extract_sentiment(self, texts: list) -> np.ndarray:
        """Predicts sentiment class probabilities using the fine-tuned sentiment model."""
        total_samples = len(texts)
        logger.info(f"[Branch B] Predicting Multilingual Sentiment ({self.sentiment_model_name}) for {total_samples} posts...")
        
        all_probabilities = []
        for i in tqdm(range(0, total_samples, self.batch_size), desc="Sentiment Inference"):
            batch_texts = texts[i : i + self.batch_size]
            logits = self.dual_encoder.forward_sentiment(batch_texts, max_length=self.max_length)
            batch_probs = self.head.predict_probabilities(logits)
            all_probabilities.append(batch_probs)
            
        return np.vstack(all_probabilities) if all_probabilities else np.empty((0, 3))

    def run(self, df: pd.DataFrame, text_column: str = "cleaned_text"):
        """Runs batch inference over the entire input DataFrame for both models."""
        texts = df[text_column].tolist()
        total_samples = len(texts)
        logger.info(f"Starting Stage 3 Multilingual Pipeline on {total_samples} posts...")
        
        # Branch A: BERTopic Sentence Embeddings
        embeddings_matrix = self.extract_embeddings(texts)
        
        # Memory optimization: empty CUDA cache between heavy model runs if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Branch B: Sentiment Analysis
        probabilities_matrix = self.extract_sentiment(texts)
        
        # Calculate Sentiment Score: S = P(positive) - P(negative)
        sentiment_scores = compute_sentiment_score(probabilities_matrix)
        
        df = df.copy()
        df["p_negative"] = probabilities_matrix[:, 0]
        df["p_neutral"] = probabilities_matrix[:, 1]
        df["p_positive"] = probabilities_matrix[:, 2]
        df["sentiment_score"] = sentiment_scores
        
        # Discrete sentiment label
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        df["sentiment_label"] = np.argmax(probabilities_matrix, axis=1)
        df["sentiment_label"] = df["sentiment_label"].map(label_map)
        
        # Save artifacts
        ensure_dir(self.output_embeddings_path)
        np.save(self.output_embeddings_path, embeddings_matrix)
        save_dataframe(df, self.output_df_path)
        
        logger.info(f"Saved Embeddings matrix with shape {embeddings_matrix.shape} to: {self.output_embeddings_path}")
        logger.info(f"Saved dataset with Sentiment Scores to: {self.output_df_path}")
        return df, embeddings_matrix

    def run_from_file(self, input_path: str):
        df = load_dataframe(input_path)
        return self.run(df)
