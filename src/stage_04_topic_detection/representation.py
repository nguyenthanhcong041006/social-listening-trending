from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from typing import Dict, Any, List

DEFAULT_SPORTS_STOPWORDS = [
    "trận", "trận đấu", "hôm nay", "video", "trực tiếp", "bóng", "sân", "full", "highlight",
    "clip", "kênh", "xem", "link", "bàn", "phút", "vs", "the", "and", "match", "game", "today"
]

def create_representation_model(config: Dict[str, Any] = None, sports_stopwords: List[str] = None):
    """
    Step 3: Topic Representation with Sports Domain Stopword Filtering
    Extracts representative topic keywords using Class-based TF-IDF (c-TF-IDF).
    Filters out common sports generic filler words to elevate unique entities, tournaments,
    players, and drama/scandal keywords.
    """
    cfg = config or {}
    stopwords = list(DEFAULT_SPORTS_STOPWORDS)
    if sports_stopwords:
        stopwords.extend(sports_stopwords)
        
    vectorizer_model = CountVectorizer(
        ngram_range=tuple(cfg.get("ngram_range", [1, 2])),
        stop_words=stopwords,
        min_df=1
    )
    ctfidf_model = ClassTfidfTransformer(
        reduce_frequent_words=True
    )
    return vectorizer_model, ctfidf_model
