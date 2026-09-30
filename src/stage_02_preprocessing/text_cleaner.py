import re
from bs4 import BeautifulSoup

def remove_html(text: str) -> str:
    """Removes HTML markup tags and unescapes HTML characters."""
    if not isinstance(text, str):
        return ""
    if "<" in text and ">" in text:
        text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    return text

def remove_urls(text: str) -> str:
    """Removes web links and URLs (http, https, www)."""
    if not isinstance(text, str):
        return ""
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.sub("", text)

def normalize_whitespaces(text: str) -> str:
    """Normalizes excessive whitespace, line breaks, and tabs into a single space."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_text(text: str, remove_url: bool = True) -> str:
    """
    Step 3 & 4: Text Cleaning & Remove URLs
    Execution order: Remove HTML -> Remove URLs -> Normalize whitespace.
    """
    text = remove_html(text)
    if remove_url:
        text = remove_urls(text)
    text = normalize_whitespaces(text)
    return text
