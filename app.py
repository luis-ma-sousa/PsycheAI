import streamlit as st
import os, json
from core.loaders import load_text_files
from core.preprocess import character_filter, chunk_text
from core.index import build_faiss
from core.retrieval import ensemble_retrieve
from core.prompts import build_prompt
from core.generation import template_fallback, hf_infer, extract_json_then_md

st.set_page_config(page_title="Psyche AI – Psychological Profiling", page_icon="🧠", layout="centered")
st.title("🧠 Psyche AI – Psychological Profiling")
st.caption("Generic RAG + optional HF API (multi-lingual) | Bring your own texts")

DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
EMB_MODEL = "intfloat/multilingual-e5-base"

st.sidebar.header("Settings")
language = st.sidebar.selectbox("Output language", ["pt", "en"], index=0)
use_hf = st.sidebar.toggle("Use Hugging Face API (if HF_API_TOKEN is set)", value=False)
hf_model = st.sidebar.text_input("HF Model", DEFAULT_MODEL, disabled=not use_hf)
k_char = st.sidebar.slider("Top-k evidence", 4, 20, 10)
k_psych = st.sidebar.slider("Top-k psychology theory", 3, 15, 6)

st.markdown("### 1) Upload your corpus (.txt or .md) – OR paste text directly")

# Tab interface for upload vs paste
tab1, tab2 = st.tabs(["📁 Upload files", "📝 Paste text"])

with tab1:
    uploaded = st.file_uploader("Upload .txt/.md", type=["txt","md"], accept_multiple_files=True)

with tab2:
    pasted_text = st.text_area(
        "Paste your text here",
        placeholder="Paste book chapters, articles, or any text corpus...",
        height=300
    )

BASE_DIR = os.path.dirname(__file__)
psych_dir = os.path.join(BASE_DIR, "knowledge", "psychology")
os.makedirs(psych_dir, exist_ok=True)

# Bootstrap minimal psych theory if empty
if not os.listdir(psych_dir):
    with open(os.path.join(psych_dir, "big_five.md"), "w", encoding="utf-8") as f:
        f.write("# Big Five (OCEAN)\n- Openness\n- Conscientiousness\n- Extraversion\n- Agreeableness\n- Neuroticism\n")
    with open(os.path.join(psych_dir, "attachment.md"), "w", encoding="utf-8") as f:
        f.write("# Attachment Styles\n- Secure\n- Anxious\n- Avoidant\n- Disorganized\n")
    with open(os.path.join(psych_dir, "coping.md"), "w", encoding="utf-8") as f:
        f.write("# Coping Strategies\n- Problem-focused\n- Emotion-focused\n- Maladaptive\n")
    with open(os.path.join(psych_dir, "defenses.md"), "w", encoding="utf-8") as f:
        f.write("# Defense Mechanisms\n- Denial, Projection, Rationalization, Displacement, Sublimation, Humor, Intellectualization.\n")

docs = []

# From uploaded files
if uploaded:
    for f in uploaded:
        text = f.read().decode("utf-8", errors="ignore")
        docs.append({"source": f.name, "text": text})

# From pasted text
if pasted_text and pasted_text.strip():
    docs.append({"source": "pasted_text", "text": pasted_text.strip()})

psych_docs = []
for name in os.listdir(psych_dir):
    if name.endswith(".md"):
        with open(os.path.join(psych_dir, name), "r", encoding="utf-8") as f:
            psych_docs.append({"source": name, "text": f.read()})

st.markdown("### 2) Define entity/character to profile")
character = st.text_input("Entity name (character/person/etc.)", "Entity X")

st.markdown("**Optional: Additional context/instructions**")
user_context = st.text_area(
    "Add custom instructions or context (optional)",
    placeholder="E.g.: Focus on relationships with other characters, analyze moral development, or examine only early chapters...",
    height=100
)

build_btn = st.button("Build indices (RAG)")

if build_btn:
    if not docs:
        st.warning("No documents uploaded or pasted. Please add text to analyze.")
    else:
        with st.spinner("Building indices..."):
            char_docs = character_filter(docs, character)
            if not char_docs:
                st.info("No direct mentions found; indexing all provided text for search anyway.")
                char_docs = docs
            char_chunks = []
            for d in char_docs:
                char_chunks.extend(chunk_text(d, chunk_size=900, overlap=200))
            psych_chunks = []
            for d in psych_docs:
                psych_chunks.extend(chunk_text(d, chunk_size=900, overlap=200))
            for i, c in enumerate(char_chunks):
                if "id" not in c: c["id"] = f"char_{i:04d}"
            for i, c in enumerate(psych_chunks):
                if "id" not in c: c["id"] = f"psy_{i:04d}"
            build_faiss(char_chunks, "character", EMB_MODEL)
            build_faiss(psych_chunks, "psych", EMB_MODEL)
        st.success("Indices built. You can now generate a profile.")

st.markdown("### 3) Generate profile")
gen_btn = st.button("Generate")

if gen_btn:
    char_query = f"{character} behaviour emotions relationships motivations internal conflict key scenes quotes descriptions"
    theory_query = "big five traits attachment styles coping mechanisms defense mechanisms psychological profiling glossary examples"

    char_hits, psych_hits = ensemble_retrieve(char_query, theory_query, EMB_MODEL, k_char, k_psych)

    with st.expander("RAG Context – Evidence"):
        for i, c in enumerate(char_hits, 1):
            st.markdown(f"**{i}. {c['id']}** (score={c['score']:.3f})")
            st.code(c["text"][:800] + (("..." if len(c["text"])>800 else "")), language="markdown")

    with st.expander("RAG Context – Psychology"):
        for i, c in enumerate(psych_hits, 1):
            st.markdown(f"**{i}. {c['id']}** (score={c['score']:.3f})")
            st.code(c["text"][:800] + (("..." if len(c["text"])>800 else "")), language="markdown")

    prompt = build_prompt(character, char_hits, psych_hits, language=language, user_context=user_context)

    if use_hf:
        try:
            with st.spinner("Calling LLM API..."):
                txt = hf_infer(prompt, model=hf_model)
            
            # Debug: show raw response
            with st.expander("🔍 Debug: Raw LLM response"):
                st.code(txt[:2000] if len(txt) > 2000 else txt, language="text")
            
            parsed = extract_json_then_md(txt)
            
            # Check if valid JSON
            if not parsed["json"] or len(parsed["json"]) == 0:
                st.warning("⚠️ LLM returned invalid JSON. Falling back to template mode.")
                use_hf = False
            else:
                st.subheader("JSON (profile)")
                st.code(json.dumps(parsed["json"], ensure_ascii=False, indent=2), language="json")
                
                # Generate markdown from JSON if missing
                if parsed.get("markdown") and parsed["markdown"].strip():
                    markdown_text = parsed["markdown"]
                else:
                    # Auto-generate from JSON
                    profile = parsed["json"]
                    markdown_text = f"""# Perfil: {profile.get('character', 'Unknown')}

**Big Five (OCEAN):**
- Abertura (O): {profile.get('big_five', {}).get('O', 'N/A')}
- Conscienciosidade (C): {profile.get('big_five', {}).get('C', 'N/A')}
- Extroversão (E): {profile.get('big_five', {}).get('E', 'N/A')}
- Amabilidade (A): {profile.get('big_five', {}).get('A', 'N/A')}
- Neuroticismo (N): {profile.get('big_five', {}).get('N', 'N/A')}

**Estilo de Apego:** {profile.get('attachment_style', 'N/A')}

**Traços Centrais:**
{chr(10).join(['- ' + t for t in profile.get('core_traits', [])])}

**Estratégias de Coping:**
{chr(10).join(['- ' + s for s in profile.get('coping_strategies', [])])}

**Arco Emocional:** {profile.get('emotional_arc', 'N/A')}

**Padrões Clínicos:**
{chr(10).join(['- ' + p for p in profile.get('clinical_patterns', [])])}

**Citações de Apoio:**
"""
                    for q in profile.get('supporting_quotes', []):
                        markdown_text += f"\n- ({q.get('chunk_id', 'N/A')}) \"{q.get('text', '')[:200]}...\"\n"
                    
                    markdown_text += f"""
**Limitações da Análise:**
{chr(10).join(['- ' + l for l in profile.get('limitations', [])])}

**Confiança:** {profile.get('confidence', 'N/A')}
"""
                
                st.subheader("Markdown report")
                st.markdown(markdown_text)
        except Exception as e:
            st.error(f"❌ HF API failed: {e}")
            st.info("ℹ️ Falling back to template mode.")
            use_hf = False

    if not use_hf:
        out = template_fallback(character, char_hits, psych_hits, language=language)
        st.subheader("JSON (profile)")
        st.code(json.dumps(out["json"], ensure_ascii=False, indent=2), language="json")
        st.subheader("Markdown report")
        st.markdown(out["markdown"])

    st.download_button("⬇️ Download prompt", prompt, file_name="prompt.txt")
