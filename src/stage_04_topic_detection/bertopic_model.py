import numpy as np
import pandas as pd
from bertopic import BERTopic
from loguru import logger
from .clustering import create_umap_model, create_hdbscan_model
from .representation import create_representation_model
from src.utils.file_io import save_dataframe, load_dataframe, ensure_dir

class TopicDetector:
    """
    Stage 4: Hierarchical Topic Detection (BERTopic) for Sports Domain
    Identifies and discovers main topics and hierarchical sub-topics (events, tournaments, drama):
    1. Embeddings (from XLM-RoBERTa Stage 3)
    2. Clustering (UMAP + HDBSCAN)
    3. Topic Representation (c-TF-IDF with Sports Stopword Filtering)
    4. Hierarchical Topic Extraction (Tree Structure: Sports -> Leagues/Clubs -> Micro-events/Drama)
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        umap_cfg = self.config.get("umap", {})
        hdbscan_cfg = self.config.get("hdbscan", {})
        repr_cfg = self.config.get("representation", {})
        sports_stopwords = self.config.get("sports_stopwords", None)
        
        self.hierarchical_cfg = self.config.get("hierarchical", {})
        self.hierarchical_enabled = self.hierarchical_cfg.get("enabled", True)
        
        self.umap_model = create_umap_model(umap_cfg)
        self.hdbscan_model = create_hdbscan_model(hdbscan_cfg)
        self.vectorizer_model, self.ctfidf_model = create_representation_model(repr_cfg, sports_stopwords=sports_stopwords)
        
        self.topic_model = BERTopic(
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=self.vectorizer_model,
            ctfidf_model=self.ctfidf_model,
            top_n_words=repr_cfg.get("top_n_words", 10),
            verbose=True
        )

    def fit_transform(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        text_column: str = "cleaned_text",
        train_ratio: float = 0.70,
        temporal_inductive: bool = True
    ):
        """
        Fits BERTopic inductively to eliminate future temporal data leakage:
        - If temporal_inductive=True and 'timestamp' is present:
          Fits UMAP + HDBSCAN + c-TF-IDF exclusively on the Train Window (first 70% timeline).
          Applies inductive .transform() to infer topics on Validation and Test windows.
        - Otherwise, runs standard batch fit_transform.
        """
        docs = df[text_column].tolist()
        n_samples = len(docs)
        logger.info(f"Starting Topic Detection with {n_samples} sports posts (Inductive Temporal: {temporal_inductive})...")
        
        has_timestamp = "timestamp" in df.columns and df["timestamp"].notna().sum() > 10
        if temporal_inductive and has_timestamp and train_ratio < 1.0:
            df_sorted_times = sorted(pd.to_datetime(df["timestamp"]).unique())
            cutoff_idx = int(len(df_sorted_times) * train_ratio)
            train_cutoff = df_sorted_times[cutoff_idx]
            is_train = pd.to_datetime(df["timestamp"]) <= train_cutoff
            train_indices = np.where(is_train.values)[0]
            val_test_indices = np.where(~is_train.values)[0]
            
            logger.info(f"Inductive Splitting: Fitting BERTopic on {len(train_indices)} Train posts (timeline <= {train_cutoff})...")
            train_docs = [docs[i] for i in train_indices]
            train_embeds = embeddings[train_indices]
            
            topics_train, probs_train = self.topic_model.fit_transform(train_docs, embeddings=train_embeds)
            
            # Reduce outliers on train set
            outlier_count = sum(1 for t in topics_train if t == -1)
            if outlier_count > 0:
                logger.info(f"Refining train topic assignments: reducing {outlier_count} outliers via c-TF-IDF...")
                try:
                    reduced_topics = self.topic_model.reduce_outliers(train_docs, topics_train, strategy="c-tf-idf")
                    self.topic_model.update_topics(train_docs, topics=reduced_topics)
                    topics_train = reduced_topics
                except Exception as e:
                    logger.warning(f"Outlier reduction skipped on train: {e}")
                    
            topics = np.zeros(n_samples, dtype=int)
            topics[train_indices] = topics_train
            probs = np.ones(n_samples, dtype=float)
            if probs_train is not None:
                probs[train_indices] = np.max(probs_train, axis=1) if len(probs_train.shape) > 1 else probs_train
            
            if len(val_test_indices) > 0:
                logger.info(f"Applying inductive .transform() on {len(val_test_indices)} Validation/Test posts...")
                val_test_docs = [docs[i] for i in val_test_indices]
                val_test_embeds = embeddings[val_test_indices]
                topics_future, probs_future = self.topic_model.transform(val_test_docs, embeddings=val_test_embeds)
                topics[val_test_indices] = topics_future
                if probs_future is not None:
                    probs[val_test_indices] = np.max(probs_future, axis=1) if len(probs_future.shape) > 1 else probs_future
                    
            topics = list(topics)
        else:
            # Fallback batch fit_transform
            topics, probs = self.topic_model.fit_transform(docs, embeddings=embeddings)
            outlier_count = sum(1 for t in topics if t == -1)
            if outlier_count > 0:
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
        
        # Extract Hierarchical Topics (Tree Structure)
        output_posts_path = self.config.get("output_posts_path", "data/04_topics/posts_with_topics.parquet")
        output_topics_path = self.config.get("output_topics_path", "data/04_topics/topic_info.csv")
        output_hierarchical_path = self.config.get("output_hierarchical_path", "data/04_topics/hierarchical_topics.csv")
        model_save_dir = self.config.get("model_save_dir", "models/bertopic/model")
        
        if self.hierarchical_enabled and len(set(topics)) > 2:
            logger.info("Extracting Hierarchical Topics Structure (Macro leagues down to Micro events/drama)...")
            try:
                hierarchical_topics = self.topic_model.hierarchical_topics(docs)
                save_dataframe(hierarchical_topics, output_hierarchical_path)
                logger.info(f"Hierarchical tree successfully extracted and saved to: {output_hierarchical_path}")
            except Exception as e:
                logger.warning(f"Hierarchical topic extraction warning: {e}")
        
        save_dataframe(df, output_posts_path)
        save_dataframe(topic_info, output_topics_path)
        
        ensure_dir(model_save_dir)
        self.topic_model.save(model_save_dir, serialization="safetensors", save_embedding_model=False)
        
        valid_topics_count = len(topic_info[topic_info["Topic"] != -1])
        logger.info(f"Hierarchical Topic Detection completed. Discovered {valid_topics_count} distinct sports sub-topics.")
        logger.info(f"Saved artifacts to: {output_posts_path} and {output_topics_path}")
        return df, topic_info
