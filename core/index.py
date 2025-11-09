import os, json
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss

STORAGE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage")
os.makedirs(STORAGE, exist_ok=True)

def build_faiss(docs: List[Dict], index_name: str, emb_model: str) -> str:
    texts = [d["text"] for d in docs]
    ids = [d["id"] for d in docs]
    model = SentenceTransformer(emb_model)
    embeds = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    dim = embeds.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeds)
    path = os.path.join(STORAGE, f"{index_name}.index")
    faiss.write_index(index, path)
    meta = {"ids": ids, "texts": texts}
    with open(os.path.join(STORAGE, f"{index_name}.meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return path

def search(query: str, index_name: str, emb_model: str, top_k: int = 5):
    idx_path = os.path.join(STORAGE, f"{index_name}.index")
    meta_path = os.path.join(STORAGE, f"{index_name}.meta.json")
    if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
        return []
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    model = SentenceTransformer(emb_model)
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    index = faiss.read_index(idx_path)
    scores, idxs = index.search(q, top_k)
    out = []
    for score, i in zip(scores[0], idxs[0]):
        if i < 0: continue
        out.append({"id": meta["ids"][i], "text": meta["texts"][i], "score": float(score)})
    return out
