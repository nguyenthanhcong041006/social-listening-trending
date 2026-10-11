import re
import emoji

# Set of multilingual negation context keywords to protect from aggressive stopword removal
NEGATION_TOKENS = {
    "not", "no", "never", "none", "neither", "nor", "hardly", "barely", "cannot", "without",
    "không", "chẳng", "chưa", "đâu", "hổng", "k", "ko", "khong"
}

# Key sports entities, slang, abbreviations to preserve during tokenization & normalization
SPORTS_SLANG_ENTITIES = {
    "mu", "mc", "chelsea", "arsenal", "liverpool", "barca", "real", "var", "c1", "c2", "epl",
    "goat", "cr7", "m10", "alcaraz", "sinner", "djokovic", "rafa", "federer", "pen", "the do",
    "the vang", "viet vi", "choke", "quay xe", "hattrick", "derby", "pep", "ten hag", "klopp"
}

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
    - Preserves negation tokens and sports-specific entity context
    """
    text = preserve_hashtags_as_words(text)
    text = handle_emojis(text, demojize=demojize)
    return text.strip()
