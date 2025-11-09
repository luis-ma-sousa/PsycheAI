from .index import search

def ensemble_retrieve(character_query: str, theory_query: str, emb_model: str, k_char: int = 8, k_psych: int = 6):
    char_hits = search(character_query, "character", emb_model, top_k=k_char)
    psych_hits = search(theory_query, "psych", emb_model, top_k=k_psych)
    return char_hits, psych_hits
