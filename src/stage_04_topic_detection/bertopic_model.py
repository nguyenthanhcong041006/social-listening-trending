import numpy as np
import pandas as pd
from bertopic import BERTopic
from loguru import logger
from .clustering import create_umap_model, create_hdbscan_model
from .representation import create_representation_model
from src.utils.file_io import save_dataframe, load_dataframe, ensure_dir

class TopicDetector:
    """
    Stage 4: Topic Detection (BERTopic)
    Identifies and discovers main topics from social media posts:
    1. Embeddings (from XLM-RoBERTa Stage 3)
    2. Clustering (UMAP + HDBSCAN)
    3. Topic Representation (c-TF-IDF)
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        umap_cfg = self.config.get("umap", {})
        hdbscan_cfg = self.config.get("hdbscan", {})
        repr_cfg = self.config.get("representation", {})
        
        self.umap_model = create_umap_model(umap_cfg)
        self.hdbscan_model = create_hdbscan_model(hdbscan_cfg)
        self.vectorizer_model, self.ctfidf_model = create_representation_model(repr_cfg)
        
        self.topic_model = BERTopic(
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=self.vectorizer_model,
            ctfidf_model=self.ctfidf_model,
            top_n_words=repr_cfg.get("top_n_words", 10),
            verbose=True
        )

    def fit_transform(self, df: pd.DataFrame, embeddings: np.ndarray, text_column: str = "cleaned_text"):
        """Fits BERTopic on posts using precomputed embeddings and assigns topic labels."""
        docs = df[text_column].tolist()
        logger.info(f"Starting Topic Detection with {len(docs)} posts...")
        
        topics, probs = self.topic_model.fit_transform(docs, embeddings=embeddings)
        
        # Step 5 (PDF Section 6): Outlier reduction and topic refinement
        outlier_count = sum(1 for t in topics if t == -1)
        if outlier_count > 0:
            logger.info(f"Refining topic assignments: reducing {outlier_count} outliers via c-TF-IDF...")
            try:
                reduced_topics = self.topic_model.reduce_outliers(docs, topics, strategy="c-tf-idf")
                self.topic_model.update_topics(docs, topics=reduced_topics)
                topics = reduced_topics
            except Exception as e:
                logger.warning(f"Outlier reduction skipped: {e}")
        
        df = df.copy()
        df["topic_id"] = topics
        if probs is not None and len(probs.shape) > 1:
            df["topic_probability"] = np.max(probs, axis=1)
        else:
            df["topic_probability"] = probs if probs is not None else 1.0

        # Retrieve topic names and keyword representations
        topic_info = self.topic_model.get_topic_info()
        topic_name_map = dict(zip(topic_info["Topic"], topic_info["Name"]))
        df["topic_name"] = df["topic_id"].map(topic_name_map)
        
        # Save results
        output_posts_path = self.config.get("output_posts_path", "data/04_topics/posts_with_topics.parquet")
        output_topics_path = self.config.get("output_topics_path", "data/04_topics/topic_info.csv")
        model_save_dir = self.config.get("model_save_dir", "models/bertopic/model")
        
        save_dataframe(df, output_posts_path)
        save_dataframe(topic_info, output_topics_path)
        
        ensure_dir(model_save_dir)
        self.topic_model.save(model_save_dir, serialization="safetensors", save_embedding_model=False)
        
        valid_topics_count = len(topic_info[topic_info["Topic"] != -1])
        logger.info(f"Topic Detection completed. Discovered {valid_topics_count} distinct topics.")
        logger.info(f"Saved artifacts to: {output_posts_path} and {output_topics_path}")
        return df, topic_info
