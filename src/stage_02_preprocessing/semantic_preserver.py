import re
import emoji

# Set of multilingual negation context keywords to protect from aggressive stopword removal
NEGATION_TOKENS = {
    "not", "no", "never", "none", "neither", "nor", "hardly", "barely", "cannot", "without",
    "không", "chẳng", "chưa", "đâu", "hổng", "k", "ko", "khong"
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

def preserve_semantic_information(text: str, demojize: bool = False) -> str:
    """
    Step 6: Preserve Semantic Information
    - Preserves hashtags as topic markers
    - Preserves meaningful sentiment emojis
    - Preserves negation tokens and semantic context
    """
    text = preserve_hashtags_as_words(text)
    text = handle_emojis(text, demojize=demojize)
    return text.strip()
