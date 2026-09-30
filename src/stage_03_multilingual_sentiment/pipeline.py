import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from .encoder import XLMRoBERTaEncoder
from .pooling import mean_pooling
from .sentiment_head import SentimentClassificationHead, compute_sentiment_score
from src.utils.file_io import save_dataframe, load_dataframe, ensure_dir

class MultilingualSentimentPipeline:
    """
    Stage 3 Pipeline:
    Concurrently extracts:
    1. Contextual Sentence Embeddings (for Stage 4 BERTopic)
    2. Sentiment Score computed via formula S = P(positive) - P(negative)
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        model_name = self.config.get("model_name", "cardiffnlp/twitter-xlm-roberta-base-sentiment")
        self.batch_size = self.config.get("batch_size", 32)
        self.max_length = self.config.get("max_length", 128)
        
        self.encoder = XLMRoBERTaEncoder(model_name=model_name)
        self.head = SentimentClassificationHead()
        
        self.output_df_path = self.config.get(
            "output_df_path", "data/03_embeddings_sentiment/posts_with_sentiment.parquet"
        )
        self.output_embeddings_path = self.config.get(
            "output_embeddings_path", "data/03_embeddings_sentiment/sentence_embeddings.npy"
        )

    def run(self, df: pd.DataFrame, text_column: str = "cleaned_text"):
        """Runs batch inference over the entire input DataFrame."""
        texts = df[text_column].tolist()
        total_samples = len(texts)
        logger.info(f"Extracting Embeddings & Sentiment for {total_samples} posts...")
        
        all_embeddings = []
        all_probabilities = []
        
        for i in tqdm(range(0, total_samples, self.batch_size), desc="XLM-RoBERTa Inference"):
            batch_texts = texts[i : i + self.batch_size]
            encoded = self.encoder.tokenize(batch_texts, max_length=self.max_length)
            outputs = self.encoder.forward(encoded)
            
            # Branch 1: Mean pooling generates sentence embeddings
            batch_embeds = mean_pooling(outputs["hidden_states"], outputs["attention_mask"])
            all_embeddings.append(batch_embeds)
            
            # Branch 2: Sentiment classification head
            batch_probs = self.head.predict_probabilities(outputs["logits"])
            all_probabilities.append(batch_probs)

        embeddings_matrix = np.vstack(all_embeddings)
        probabilities_matrix = np.vstack(all_probabilities)
        
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
