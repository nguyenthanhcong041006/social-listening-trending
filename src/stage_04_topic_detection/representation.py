from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from bertopic.vectorizers import ClassTfidfTransformer
from typing import Dict, Any, List

# Comprehensive sports domain filler words in English social media discourse
DEFAULT_SPORTS_STOPWORDS = [
    "game", "match", "today", "tonight", "live", "stream", "watch", "link",
    "highlights", "highlight", "video", "clip", "full", "vs", "season",
    "action", "quarter", "halftime", "final", "score", "scores", "update", "updates",
    "check", "post", "posts", "read", "click", "via", "news", "breaking", "official",
    "schedule", "broadcast", "coverage", "day", "night", "week", "fans", "fan",
    "team", "teams", "player", "players", "win", "wins", "loss", "losses",
    "playing", "played", "play", "year", "years", "time", "times"
]

def create_representation_model(config: Dict[str, Any] = None, sports_stopwords: List[str] = None):
    """
    Step 3: Topic Representation with Sports Domain Stopword Filtering
    Extracts representative topic keywords using Class-based TF-IDF (c-TF-IDF).
    Combines standard English stopwords with sports generic filler words to elevate unique entities,
    tournaments, clubs, athletes, and drama/rivalry keywords.
    """
    cfg = config or {}
    
    # Combine standard English stopwords with sports domain filler words
    combined_stopwords = set(ENGLISH_STOP_WORDS).union(DEFAULT_SPORTS_STOPWORDS)
    if sports_stopwords:
        combined_stopwords.update(sports_stopwords)
        
    vectorizer_model = CountVectorizer(
        ngram_range=tuple(cfg.get("ngram_range", [1, 2])),
        stop_words=list(combined_stopwords),
        min_df=1
    )
    ctfidf_model = ClassTfidfTransformer(
        reduce_frequent_words=True
    )
    return vectorizer_model, ctfidf_model
