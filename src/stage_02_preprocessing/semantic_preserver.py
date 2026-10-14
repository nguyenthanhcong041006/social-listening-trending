import re
import emoji

# Set of multilingual negation context keywords to protect from aggressive stopword removal
NEGATION_TOKENS = {
    "not", "no", "never", "none", "neither", "nor", "hardly", "barely", "cannot", "without",
    "không", "chẳng", "chưa", "đâu", "hổng", "k", "ko", "khong"
}

# Key sports entities, slang, abbreviations to preserve during tokenization & normalization
SPORTS_CANONICAL_ACRONYMS = {
    "mu": "MU",
    "mc": "MC",
    "var": "VAR",
    "c1": "C1",
    "c2": "C2",
    "epl": "EPL",
    "goat": "GOAT",
    "cr7": "CR7",
    "m10": "M10",
    "nba": "NBA",
    "f1": "F1",
    "ucl": "UCL",
    "uel": "UEL"
}

SPORTS_SLANG_ENTITIES = {
    "chelsea", "arsenal", "liverpool", "barca", "real", "alcaraz", "sinner", "djokovic",
    "rafa", "federer", "pen", "the do", "the vang", "viet vi", "choke", "quay xe",
    "hattrick", "derby", "pep", "ten hag", "klopp", "haaland", "mbappe", "vinicius"
}

def protect_sports_entities(text: str) -> str:
    """
    Standardizes and protects sports acronyms so they retain canonical uppercase
    (e.g., 'mu' -> 'MU', 'var' -> 'VAR') for Cased XLM-RoBERTa tokenization.
    """
    if not isinstance(text, str):
        return ""
    for lower_term, canonical in SPORTS_CANONICAL_ACRONYMS.items():
        pattern = re.compile(r"\b" + re.escape(lower_term) + r"\b", re.IGNORECASE)
        text = pattern.sub(canonical, text)
    return text

def compute_uppercase_ratio(text: str) -> float:
    """
    Computes the proportion of uppercase alphabetic characters in a text.
    Acts as an emotional arousal and controversy indicator (shouting, outrage, excitement).
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return float(sum(1 for c in letters if c.isupper()) / len(letters))

def preserve_hashtags_as_words(text: str) -> str:
    """
    Preserves hashtags as meaningful semantic tokens with clean boundary spacing.
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r"(#\w+)", r" \1 ", text)

def handle_emojis(text: str, demojize: bool = False) -> str:
    """
    Preserves emotionally informative emojis.
    If demojize=True, converts emojis to text descriptions (e.g. ❤️ -> :red_heart:).
    If demojize=False, keeps raw Unicode emojis for transformer models that natively support them.
    """
    if not isinstance(text, str):
        return ""
    if demojize:
        return emoji.demojize(text, language="en")
    return text

def preserve_semantic_information(text: str, demojize: bool = False, preserve_sports_slang: bool = True) -> str:
    """
    Step 6: Preserve Semantic Information
    - Preserves hashtags as topic markers
    - Preserves meaningful sentiment emojis
    - Standardizes sports entities into canonical cased format (MU, VAR, C1)
    - Retains emotional uppercase signals
    """
    text = preserve_hashtags_as_words(text)
    text = handle_emojis(text, demojize=demojize)
    if preserve_sports_slang:
        text = protect_sports_entities(text)
    return text.strip()
