from typing import List, Dict
from .utils import normalize_text, contains_any

def make_aliases(name: str) -> List[str]:
    name_low = name.lower().strip()
    parts = name_low.split()
    aliases = {name_low}
    if len(parts) >= 2:
        aliases.add(parts[0])
        aliases.add(parts[-1])
    return sorted(aliases)

def character_filter(docs: List[Dict], character: str) -> List[Dict]:
    aliases = make_aliases(character)
    out = []
    for d in docs:
        if contains_any(d["text"], aliases):
            out.append(d)
    return out

def chunk_text(doc: Dict, chunk_size: int = 900, overlap: int = 200) -> List[Dict]:
    words = doc["text"].split()
    chunks = []
    i = 0
    idx = 0
    while i < len(words):
        piece = " ".join(words[i:i+chunk_size])
        chunks.append({
            "id": f'{doc["source"]}#chunk{idx:04d}',
            "source": doc["source"],
            "text": piece
        })
        i += (chunk_size - overlap)
        idx += 1
    return chunks
