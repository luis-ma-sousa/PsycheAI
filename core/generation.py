import os, json, re
from typing import Dict, Any
from pydantic import BaseModel, Field


class Profile(BaseModel):
    character: str
    big_five: Dict[str, float] = Field(default_factory=dict)
    attachment_style: str = ""
    core_traits: list = Field(default_factory=list)
    coping_strategies: list = Field(default_factory=list)
    emotional_arc: str = ""
    clinical_patterns: list = Field(default_factory=list)
    supporting_quotes: list = Field(default_factory=list)
    limitations: list = Field(default_factory=list)
    confidence: float = 0.7


def template_fallback(character: str, char_chunks: list, psych_chunks: list, language: str = "pt") -> Dict[str, Any]:
    """
    Geração determinística de emergência (sem LLM), usando os chunks recuperados.
    """
    quotes = []
    for c in char_chunks[:4]:
        quotes.append({
            "text": c["text"][:160] + ("..." if len(c["text"]) > 160 else ""),
            "source": "corpus",
            "chunk_id": c["id"]
        })
    base = {
        "character": character,
        "big_five": {"O": 0.7, "C": 0.6, "E": 0.4, "A": 0.6, "N": 0.5},
        "attachment_style": "mixed/uncertain",
        "core_traits": ["goal-directed", "emotionally guarded"],
        "coping_strategies": ["planning", "information-seeking"],
        "emotional_arc": "fluctuating affect with moments of restraint",
        "clinical_patterns": ["speculative defensive sarcasm"],
        "supporting_quotes": quotes,
        "limitations": ["template fallback; no LLM generation"],
        "confidence": 0.6
    }
    md = f"""# Perfil: {character}
- Traços centrais: {", ".join(base["core_traits"])}
- Estilo de apego: {base["attachment_style"]}
- Arco emocional: {base["emotional_arc"]}

### Citações (evidência)
""" + "\n".join([f'- ({q["chunk_id"]}) {q["text"]}' for q in quotes])
    return {"json": base, "markdown": md}


def hf_infer(prompt: str, model: str, temperature: float = 0.2, max_new_tokens: int = 800) -> str:
    """
    Usa o SDK oficial da Hugging Face.
    Para Llama 3 (alguns providers), a tarefa suportada é 'conversational' → chat_completion().
    Se chat não estiver disponível noutro modelo, faz fallback para text_generation().
    """
    from huggingface_hub import InferenceClient
    from huggingface_hub.errors import HfHubHTTPError

    token = os.getenv("HF_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_API_TOKEN not set")

    client = InferenceClient(model=model, token=token)

    # 1) Tentar chat (compatível com Llama 3 via providers que exigem 'conversational')
    try:
        resp = client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert psychologist. Follow the instructions precisely. "
                        "Return output in TWO parts: "
                        "(1) a valid JSON strictly following the provided schema; "
                        "(2) then the line --- on its own line; "
                        "(3) then a concise Markdown summary with citations (chunk_id)."
                    )
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_new_tokens,    # parâmetro esperado pelo chat_completion
            temperature=temperature,
            top_p=0.95,
            stream=False,
        )
        # Extrair conteúdo de forma robusta (diferentes providers = diferentes estruturas)
        choice = resp.choices[0]
        content = ""
        if hasattr(choice, "message") and isinstance(choice.message, dict):
            content = choice.message.get("content", "")
        elif hasattr(choice, "message") and hasattr(choice.message, "content"):
            content = choice.message.content
        elif isinstance(choice, dict) and "message" in choice:
            msg = choice["message"]
            content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
        else:
            content = str(choice)

        if isinstance(content, list):  # alguns providers devolvem lista de partes
            return "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in content])
        return content
    except HfHubHTTPError as e_chat:
        # 2) Fallback para text_generation (ex.: Mistral, Zephyr, etc.)
        try:
            return client.text_generation(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                stream=False,
            )
        except HfHubHTTPError as e_txt:
            raise RuntimeError(f"Hugging Face inference error (chat/text): {e_chat} / {e_txt}") from e_txt


def _clean_json_str(j_text: str) -> str:
    """
    Limpa ruído comum em respostas LLM antes de json.loads:
    - remove comentários tipo //...
    - remove vírgulas a mais antes de } ou ]
    - extrai conteúdo de blocos ```json ... ```
    """
    # remove comentários linha
    j_text = re.sub(r'//.*', '', j_text)
    # remove vírgula à esquerda de } ou ]
    j_text = re.sub(r',(\s*[}\]])', r'\1', j_text)
    # extrai bloco json dentro de ``` ```
    m = re.search(r'```(?:json)?\s*(\{.*\})\s*```', j_text, re.DOTALL)
    if m:
        j_text = m.group(1)
    return j_text.strip()


def extract_json_then_md(text: str) -> Dict[str, Any]:
    """
    Extrai JSON (primeira estrutura {...}) e depois tenta obter Markdown do resto.
    Se o prompt tiver imposto o separador '---', também é suportado.
    """
    if "\n---\n" in text:
        json_part, md_part = text.split("\n---\n", 1)
    else:
        json_part, md_part = text, ""

    start = json_part.find("{")
    end = json_part.rfind("}")
    j = {}
    if start >= 0 and end > start:
        j_text = _clean_json_str(json_part[start:end + 1])
        try:
            j = json.loads(j_text)
        except Exception:
            # tenta novamente procurando bloco de código explícito
            code_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', json_part, re.DOTALL)
            if code_block:
                j_text = _clean_json_str(code_block.group(1))
                try:
                    j = json.loads(j_text)
                except Exception:
                    j = {}

    if not md_part:
        md_part = json_part[end + 1:].strip() if end > start else ""
    return {"json": j, "markdown": md_part}


def json_profile_to_markdown(profile: Dict[str, Any]) -> str:
    """
    Gera Markdown sólido a partir do JSON quando o modelo não devolve a parte de texto.
    """
    def list_block(items, title):
        if not items:
            return f"**{title}:** _n/a_\n"
        return f"**{title}:**\n" + "\n".join(f"- {it}" for it in items) + "\n"

    bf = profile.get("big_five", {}) or {}

    def num(x):
        try:
            return float(x)
        except Exception:
            return x if x is not None else "n/a"

    md = [f"# Perfil: {profile.get('character','Desconhecido')}"]

    md.append("\n**Big Five (OCEAN):**")
    md.append(f"- Abertura (O): {num(bf.get('O','n/a'))}")
    md.append(f"- Conscienciosidade (C): {num(bf.get('C','n/a'))}")
    md.append(f"- Extroversão (E): {num(bf.get('E','n/a'))}")
    md.append(f"- Amabilidade (A): {num(bf.get('A','n/a'))}")
    md.append(f"- Neuroticismo (N): {num(bf.get('N','n/a'))}\n")

    md.append(f"**Estilo de Apego:** {profile.get('attachment_style','n/a')}\n")
    md.append(list_block(profile.get("core_traits", []), "Traços Centrais"))
    md.append(list_block(profile.get("coping_strategies", []), "Estratégias de Coping"))

    md.append(f"**Arco Emocional:** {profile.get('emotional_arc','n/a')}\n")
    md.append(list_block(profile.get("clinical_patterns", []), "Padrões Clínicos"))

    quotes = profile.get("supporting_quotes", []) or []
    if quotes:
        md.append("**Citações de Apoio:**")
        for q in quotes:
            txt = (q.get("text", "") or "")[:200]
            md.append(f'- ({q.get("chunk_id","n/a")}) "{txt}..."')
        md.append("")
    else:
        md.append("**Citações de Apoio:** _n/a_\n")

    lim = profile.get("limitations", []) or []
    if lim:
        md.append("**Limitações da Análise:**")
        md.extend(f"- {l}" for l in lim)
        md.append("")
    else:
        md.append("**Limitações da Análise:** _n/a_\n")

    md.append(f"**Confiança:** {profile.get('confidence','n/a')}")
    return "\n".join(md).strip()
