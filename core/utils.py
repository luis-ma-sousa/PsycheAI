import re

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def contains_any(s: str, keywords):
    s_low = (s or "").lower()
    return any(k.lower() in s_low for k in keywords)
