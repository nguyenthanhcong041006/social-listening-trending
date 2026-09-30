from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from typing import Dict, Any

def create_representation_model(config: Dict[str, Any] = None):
    """
    Step 3: Topic Representation
    Extracts representative topic keywords using Class-based TF-IDF (c-TF-IDF).
    Formula:
        W_{t, c} = ||tf_{t, c}|| * log(1 + A / tf_t)
    where tf_{t, c} is term frequency in cluster c, and A is average cluster word count.
    """
    cfg = config or {}
    vectorizer_model = CountVectorizer(
        ngram_range=tuple(cfg.get("ngram_range", [1, 2])),
        stop_words="english",
        min_df=2
    )
    ctfidf_model = ClassTfidfTransformer(
        reduce_frequent_words=True
    )
    return vectorizer_model, ctfidf_model
