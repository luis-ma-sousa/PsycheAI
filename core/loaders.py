import os
from typing import List, Dict
from .utils import normalize_text

def load_text_files(paths: List[str]) -> List[Dict]:
    docs = []
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        docs.append({"source": os.path.basename(path), "text": normalize_text(text)})
    return docs
