import re
import unicodedata
import ftfy

def normalize_unicode(text: str, form: str = "NFKC") -> str:
    """
    Standardizes Unicode encoding (fixes mojibake, combining characters, and multi-script symbols).
    """
    if not isinstance(text, str):
        return ""
    text = ftfy.fix_text(text)
    return unicodedata.normalize(form, text)

def normalize_case(text: str) -> str:
    """
    Converts text to uniform lowercase.
    """
    if not isinstance(text, str):
        return ""
    return text.lower()

def handle_repeated_characters(text: str, max_repeats: int = 2) -> str:
    """
    Collapses excessive repeated characters (e.g. 'yessssss' -> 'yess', 'soooo' -> 'soo').
    Prevents out-of-vocabulary (OOV) expansion in language model tokenization.
    """
    if not isinstance(text, str):
        return ""
    pattern = re.compile(r"(.)\1{" + str(max_repeats) + r",}")
    return pattern.sub(r"\1" * max_repeats, text)

def normalize_text(text: str, form: str = "NFKC", lowercase: bool = True, max_repeats: int = 2) -> str:
    """
    Step 5: Text Normalization
    Unicode Normalization -> Case Normalization -> Repeated-Character Handling.
    """
    text = normalize_unicode(text, form=form)
    if lowercase:
        text = normalize_case(text)
    text = handle_repeated_characters(text, max_repeats=max_repeats)
    return text
