# 🧠 Psyche AI – Understanding Minds through Text with Llama-3 and RAG

**Psyche AI** is a proof-of-concept (POC) demonstrating how modern language models and retrieval-augmented generation (RAG) can be applied to **psychological analysis grounded in evidence and theory**.

By combining cognitive science, NLP, and interpretable AI, Psyche AI shows how a future system could support psychologists, educators, and researchers by summarising personality structure, emotional patterns, and coping strategies from any written material — from **literature and case notes to transcripts and interviews**.

While this project is not a diagnostic tool, it illustrates the potential of fine-tuned LLMs and DSM-5-aligned frameworks to enable **transparent, theory-driven AI agents for psychological reasoning**.

---

## 🎯 What It Does

Psyche AI analyzes entities (characters, people, etc.) by:

1. **Retrieving evidence** from your corpus (books, articles, documents)
2. **Grounding analysis** in psychological theory (Big Five, attachment styles, coping mechanisms)
3. **Generating structured profiles** using LLMs (Llama-3 via HuggingFace API)
4. **Providing citations** for transparency and verification

**Key Feature:** Evidence-based analysis with full citation traceability

---

## ✨ Features

- ✅ **RAG Pipeline:** FAISS vector search + semantic retrieval
- ✅ **Dual Indexing:** Separate indices for evidence (corpus) and theory (psychology)
- ✅ **Structured Output:** Validated JSON (Pydantic) + human-readable Markdown
- ✅ **Multi-lingual:** Portuguese (PT-PT) and English support
- ✅ **Flexible Input:** Upload files (.txt, .md) OR paste text directly
- ✅ **Transparent:** Shows retrieved chunks and confidence scores
- ✅ **Robust:** Template fallback when API unavailable
- ✅ **Production-Ready:** Error handling, validation, clean architecture

---

## 🏗️ Architecture
```
┌─────────────────┐
│   User Input    │ (Upload text / Paste corpus)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │ (Chunk text, filter by character)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FAISS Indexing │ (Create vector embeddings)
├─────────────────┤
│ • Character     │ (Evidence from corpus)
│ • Psychology    │ (Theory: Big Five, etc.)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   RAG Retrieval │ (Semantic search: top-k chunks)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Prompt Builder  │ (Evidence + Theory → LLM prompt)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Generation │ (Llama-3 via HuggingFace API)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  JSON + MD      │ (Structured profile + report)
└─────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- HuggingFace API token (optional, for LLM generation)

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/psyche-ai.git
cd psyche-ai

# Install dependencies
pip install -r requirements.txt
```

### Run
```bash
# Set HuggingFace token (optional)
export HF_API_TOKEN=your_token_here

# Launch app
streamlit run app.py
```

Open browser at `http://localhost:8501`

---

## 📖 Usage

### Step 1: Upload Corpus
- Upload `.txt` or `.md` files (books, articles, documents)
- **OR** paste text directly into text area

### Step 2: Define Entity
- Enter character/person name (e.g., "Harry Potter", "principezinho", "Frodo", etc)
- Optionally add custom instructions (e.g., "Focus on moral development")

### Step 3: Build Indices
- Click **"Build indices (RAG)"**
- System chunks text and creates vector indices

### Step 4: Generate Profile
- Click **"Generate"**
- View structured JSON profile + Markdown report
- Inspect retrieved evidence chunks (expandable sections)

---

## 📚 Dataset & Licensing

Psyche AI uses text corpora for psychological analysis via RAG.  
Only **public-domain** or **user-provided** materials are included or processed.

### 📖 Example Sources
- **Public domain:** *O Principezinho* (Antoine de Saint-Exupéry, 1943) — freely redistributable under EU copyright law.  
- **Publicly available datasets (not redistributed here):**  
  - [Harry Potter Books Dataset (Kaggle)](https://www.kaggle.com/datasets/shubhammaindola/harry-potter-books)  
  - [Movie Scripts Corpus (Kaggle)](https://www.kaggle.com/datasets/gufukuro/movie-scripts-corpus)

> ⚖️ **Disclaimer:** All copyrighted materials remain property of their respective rights holders.  
> Psyche AI does not host or distribute such content; users are responsible for ensuring legal compliance when uploading or processing external texts.  
> These dataset links are provided **for educational and research purposes only**.


---

## 🧪 Example Output

### Input
- **Corpus:** *O Principezinho* (Antoine de Saint-Exupéry)
- **Entity:** "principezinho"

### Output (JSON)
```json
{
  "character": "O Principezinho",
  "big_five": {
    "O": "0.8",
    "C": "0.7",
    "E": "0.6",
    "A": "0.9",
    "N": "0.4"
  },
  "attachment_style": "Secure",
  "core_traits": [
    "Curioso",
    "Consciencioso",
    "Amável",
    "Estável emocionalmente"
  ],
  "coping_strategies": [
    "Problem-focused",
    "Emotion-focused"
  ],
  "emotional_arc": "Estável",
  "clinical_patterns": [
    "Busca por significado",
    "Capacidade de empatia"
  ],
  "supporting_quotes": [
    {
      "text": "Os homens do teu planeta cultivam cinco mil rosas num mesmo jardim... e não encontram o que procuram.",
      "source": "O Principezinho",
      "chunk_id": "chunk0008"
    },
    {
      "text": "É preciso buscar com o coração.",
      "source": "O Principezinho",
      "chunk_id": "chunk0008"
    }
  ],
  "limitations": [
    "Dificuldade em lidar com a complexidade",
    "Tendência a idealizar"
  ],
  "confidence": "0.9"
}
```

### Output (Markdown)
```markdown

O Principezinho é um personagem curioso e conscientioso, com uma tendência a buscar significado e a empatizar com os outros. Ele é amável e estável emocionalmente, com uma capacidade de lidar com as emoções de forma saudável. No entanto, ele também tem dificuldade em lidar com a complexidade e tendência a idealizar.

[chunk0008] "Os homens do teu planeta cultivam cinco mil rosas num mesmo jardim... e não encontram o que procuram." - O Principezinho

[chunk0008] "É preciso buscar com o coração." - O Principezinho

[chunk0001] "Não tem importância. Desenha-me um carneiro." - O Principezinho

[chunk0008] "É preciso buscar com o coração." - O Principezinho
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Embeddings** | SentenceTransformers (multilingual-e5-base) |
| **Vector Search** | FAISS (Facebook AI Similarity Search) |
| **LLM** | Llama-3-8B-Instruct (via HuggingFace API) |
| **Validation** | Pydantic |
| **UI** | Streamlit |
| **Language** | Python 3.8+ |

---

## 📁 Project Structure
```
PsycheAI/
├── app.py                            # Streamlit UI (main entry point)
├── requirements.txt                  # Python dependencies
├── core/                             # Core logic modules
│ ├── init.py
│ ├── generation.py                   # LLM integration & fallback
│ ├── prompts.py                      # Prompt engineering
│ ├── index.py                        # FAISS vector indexing
│ ├── loaders.py                      # File loading utilities
│ ├── preprocess.py                   # Text chunking & filtering
│ ├── retrieval.py                    # RAG retrieval logic
│ └── utils.py                        # Helper functions
├── example_input/                    # Public-domain example corpus
│ ├── 1943_O Principezinho_PT-PT_Antoine de Saint-Exupéry.txt
│ └── README.md                       # Source and license information
├── knowledge/                        # Psychology knowledge base
│ ├── init.py
│ └── psychology/
│ ├── big_five.md
│ ├── attachment.md
│ ├── coping.md
│ └── defenses.md
└── storage/                          # Generated FAISS indices (runtime)
```

---

## ⚙️ Configuration

### Settings (Sidebar)

- **Output Language:** Portuguese (PT-PT) / English
- **Use HuggingFace API:** Toggle LLM generation (requires token)
- **HF Model:** Specify model (default: Llama-3-8B-Instruct)
- **Top-k Evidence:** Number of corpus chunks to retrieve (4-20)
- **Top-k Psychology:** Number of theory chunks to retrieve (3-15)

---

## 🧠 Psychology Knowledge Base

System includes minimal psychology theory files:

- **Big Five (OCEAN):** Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism
- **Attachment Styles:** Secure, Anxious, Avoidant, Disorganized
- **Coping Strategies:** Problem-focused, Emotion-focused, Maladaptive
- **Defense Mechanisms:** Denial, Projection, Rationalization, etc.

**Extensible:** Add your own `.md` files to `knowledge/psychology/`

---

## 🔒 Privacy & Data

- **No data collection:** All processing happens locally
- **No external logging:** Corpus never leaves your machine
- **HuggingFace API:** Only prompts sent (if API enabled)
- **Storage:** Indices saved to `storage/` directory (delete to clear)

---

## 🐛 Troubleshooting

### "No documents uploaded or pasted"
- Ensure files are `.txt` or `.md` format
- Check text area is not empty

### "HF API failed"
- Verify `HF_API_TOKEN` is set correctly
- Check API rate limits
- System will fallback to template mode automatically

### "LLM returned invalid JSON"
- Template fallback will activate automatically
- Check debug expander for raw LLM output

---

## 🎓 Use Cases

- **Literary Analysis:** Profile fictional characters with evidence
- **Historical Research:** Analyze historical figures from primary sources
- **Psychology Education:** Demonstrate personality theory application
- **Content Analysis:** Extract psychological patterns from text
- **Research Tool:** Ground qualitative analysis in theory

---

## 🚧 Limitations

- **Corpus-dependent:** Quality depends on input text richness
- **Not clinical diagnosis:** Educational/analytical tool only
- **LLM variability:** Different models produce different analyses
- **Language:** Optimized for Portuguese (PT-PT) and English
- **Context limits:** Chunks may miss broader narrative arcs

---

## 🔮 Future Enhancements

- [ ] Support for PDF uploads
- [ ] Multi-character comparison mode
- [ ] Temporal analysis (character development over time)
- [ ] Custom psychology theory upload
- [ ] Advanced visualization (trait radar charts)
- [ ] Fine-tuning on domain-specific data

---

## 📊 Technical Details

### RAG Pipeline Specifics

**Chunking Strategy:**
- Chunk size: 900 words
- Overlap: 200 words
- Why overlap? Prevents context loss at boundaries

**Embedding Model:**
- `intfloat/multilingual-e5-base` (768 dimensions)
- Supports 100+ languages
- Normalized embeddings (cosine similarity via inner product)

**Vector Search:**
- FAISS IndexFlatIP (exact search)
- Cosine similarity scoring
- Top-k retrieval (configurable)

**Prompt Engineering:**
- PT-PT language specification (avoids Brazilian Portuguese)
- JSON schema enforcement
- Evidence + theory grounding
- User instruction incorporation

**Output Validation:**
- Pydantic schema validation
- JSON cleaning (removes LLM artifacts: comments, trailing commas)
- Markdown fallback generation
- Confidence scoring

---

## 👨‍💻 Author

Luís Sousa — [LinkedIn](https://www.linkedin.com/in/luis-ma-sousa31) | [GitHub](https://github.com/luismasousa)

---

## 🔗 Related Projects

- **[MentalHealthLog](https://github.com/luis-ma-sousa/MentalHealthLog)** — Full-stack mental health tracker
