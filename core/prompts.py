PROFILE_JSON_SCHEMA = {
  "character": "string",
  "big_five": {"O": "0-1", "C": "0-1", "E": "0-1", "A": "0-1", "N": "0-1"},
  "attachment_style": "string",
  "core_traits": ["string", "..."],
  "coping_strategies": ["string", "..."],
  "emotional_arc": "string",
  "clinical_patterns": ["string", "..."],
  "supporting_quotes": [ {"text":"string","source":"string","chunk_id":"string"} ],
  "limitations": ["string", "..."],
  "confidence": "0-1"
}

def build_prompt(character: str, char_chunks: list, psych_chunks: list, language: str = "pt", user_context: str = "") -> str:
    char_ctx = "\n\n".join([f"[{c['id']}] {c['text'][:800]}" for c in char_chunks])
    psy_ctx = "\n\n".join([f"[{c['id']}] {c['text'][:800]}" for c in psych_chunks])

    # User context section (if provided)
    if user_context and user_context.strip():
        user_section_pt = f"""

INSTRUÇÕES ADICIONAIS DO UTILIZADOR:
{user_context.strip()}

(Incorpora estas instruções na análise, mas mantém o formato JSON + Markdown.)
"""
        user_section_en = f"""

ADDITIONAL USER INSTRUCTIONS:
{user_context.strip()}

(Incorporate these instructions in the analysis, but maintain JSON + Markdown format.)
"""
    else:
        user_section_pt = user_section_en = ""

    instr_pt = f"""
Tu és um psicólogo português a perfilar uma entidade (personagem, figura histórica, etc.).

INSTRUÇÕES CRÍTICAS:
- Usa APENAS as EVIDÊNCIAS e TEORIA fornecidas abaixo
- Usa português europeu (PT-PT), não brasileiro
- Não faças diagnóstico clínico formal; descreve traços e padrões comportamentais
- Primeiro devolve um JSON válido com as chaves do schema
- Depois adiciona um resumo em Markdown com citações (usar chunk_id)

LINGUAGEM PT-PT:
- Usa "curioso" (não "amoroso")
- Usa "consciencioso" (não "consciente")  
- Usa "extrovertido" (grafia PT-PT)
- Usa "amável" ou "simpático" (não tradução literal de "agreeable")
- Usa "estável emocionalmente" vs "neurótico"

Entidade a analisar: {character}

JSON_SCHEMA (devolve exatamente estas chaves):
{PROFILE_JSON_SCHEMA}

EVIDÊNCIAS (citações do corpus):
{char_ctx}

TEORIA (conceitos psicológicos):
{psy_ctx}
{user_section_pt}

FORMATO DE RESPOSTA:
1. Primeiro: JSON válido com todas as chaves do schema
2. Depois: Breve resumo em Markdown formatado, incluindo citações com [chunk_id]

Começa já com o JSON:
"""

    instr_en = f"""
You are a psychologist profiling an entity (character, historical figure, etc.).

CRITICAL INSTRUCTIONS:
- Use ONLY the EVIDENCE and THEORY provided below
- Do not make formal clinical diagnoses; describe behavioral patterns and traits
- First return a valid JSON with the schema keys
- Then add a Markdown summary with citations (using chunk_id)

Entity to analyze: {character}

JSON_SCHEMA (return exactly these keys):
{PROFILE_JSON_SCHEMA}

EVIDENCE (corpus quotes):
{char_ctx}

THEORY (psychological concepts):
{psy_ctx}
{user_section_en}

RESPONSE FORMAT:
1. First: Valid JSON with all schema keys
2. Then: Brief Markdown summary, including citations with [chunk_id]

Start with the JSON now:
"""
    return instr_pt if language.lower().startswith("pt") else instr_en
