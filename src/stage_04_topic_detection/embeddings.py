import numpy as np
from loguru import logger

def load_or_validate_embeddings(embeddings_path: str, expected_length: int = None) -> np.ndarray:
    """
    Step 1: Embeddings
    Loads Contextual Sentence Embeddings from Stage 3 (XLM-RoBERTa).
    Ensures the number of embedding vectors matches the number of post documents.
    """
    logger.info(f"Loading sentence embeddings from: {embeddings_path}")
    embeddings = np.load(embeddings_path)
    
    if expected_length is not None and len(embeddings) != expected_length:
        raise ValueError(
            f"Embedding matrix row count ({len(embeddings)}) does not match post count ({expected_length})!"
        )
    
    logger.info(f"Successfully loaded Embeddings matrix with shape: {embeddings.shape}")
    return embeddings
