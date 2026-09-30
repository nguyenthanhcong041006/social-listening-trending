from .deduplicator import remove_duplicates
from .spam_filter import filter_spam_and_bots
from .text_cleaner import clean_text, remove_html, remove_urls, normalize_whitespaces
from .text_normalizer import normalize_text, normalize_unicode, normalize_case, handle_repeated_characters
from .semantic_preserver import preserve_semantic_information
from .pipeline import PreprocessingPipeline

__all__ = [
    "remove_duplicates",
    "filter_spam_and_bots",
    "clean_text",
    "remove_html",
    "remove_urls",
    "normalize_whitespaces",
    "normalize_text",
    "normalize_unicode",
    "normalize_case",
    "handle_repeated_characters",
    "preserve_semantic_information",
    "PreprocessingPipeline",
]
