"""English-language filter. The clustering pass (scripts/04) revealed substantial
non-English traffic (German/Japanese/French/Spanish) and off-topic promotional tweets
mixed into the reconstructed pairs. We scope this assignment to English-language,
support-relevant conversations only -- documented explicitly in DECISIONS.md rather than
silently dropped."""
from langdetect import detect, DetectorFactory, LangDetectException
DetectorFactory.seed = 42

def is_english(text: str) -> bool:
    if not text or len(text.strip()) < 3:
        return False
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False
