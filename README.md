# 🏨 HotelBot Élite — Hybrid RAG Concierge

A fully offline AI concierge for luxury hotels built with a retrieval-first hybrid RAG architecture. Designed for environments where **privacy, latency, and response reliability matter more than raw generative capability**.

> **Design Philosophy:** The system is architected so that core concierge functionality remains fully usable even if the LLM is disabled. Deterministic FAQ retrieval always takes priority — the LLM is a fallback, not the foundation.

---

## 🚀 Features

- 🔍 **Hybrid RAG pipeline** — semantic retrieval first, local LLM only when confidence is insufficient
- 🎯 **3-tier confidence routing** based on calibrated similarity thresholds:
  - High confidence → direct deterministic FAQ answer
  - Medium confidence → hedged FAQ answer with clarification cue
  - Low confidence → constrained LLM generation
- 🧠 **FAISS HNSW index** with normalized dense embeddings for fast approximate nearest-neighbor search
- ⚡ **Multi-layer caching** — FAQ result cache + LLM response cache for consistent latency
- 📊 **Live metrics sidebar** — total queries, FAQ hit rate, LLM fallback count, avg latency, avg similarity
- 🔒 **Fully offline** — no cloud calls, no API keys, no data leakage
- 💬 **120+ FAQ knowledge base** covering rooms, vehicles, dining, security, amenities, pets, and local Indore experiences
- 🏨 **Luxury Streamlit UI** with warm chat bubbles and hotel branding

---

## 🏗 Architecture

```
Guest Query
     ↓
Sentence Embedding (MPNet / MiniLM)
     ↓
FAISS HNSW Semantic Search (Top-K=3, normalized dot product)
     ↓
Confidence Scoring
     ↓
┌──────────────────────────────────────────────┐
│  sim > 0.75  →  Direct FAQ Answer            │
│  0.60–0.75   →  Hedged FAQ + Clarification   │
│  sim < 0.60  →  Local LLM Fallback           │
└──────────────────────────────────────────────┘
     ↓
Multi-layer Cache (FAQ cache + GPT cache)
     ↓
Streamlit Chat UI + Metrics Sidebar
```

---

## 🧪 Model Experimentation

This project was built through deliberate experimentation across multiple LLMs and embedding models. The final configuration reflects what worked best for a luxury hotel persona on local CPU hardware.

### Embedding Models Tested

| Model | Observations |
|---|---|
| `all-MiniLM-L6-v2` | Fast, lightweight, good for most FAQ queries. Similarity scores work well with 0.7 threshold. |
| `all-mpnet-base-v2` | Better semantic accuracy on paraphrased queries, slower, requires recalibrated thresholds. **Final choice.** |

### LLMs Tested (via GPT4All)

| Model | Observations |
|---|---|
| **Snoozy 13B** | Best response quality and tone. Naturally warm and conversational — ideal for luxury hotel persona. Slow on CPU (~120–150s) but worth it for quality. **Final choice.** |
| Mistral 7B | Best speed/quality tradeoff. Slightly more clinical tone. Good alternative if Snoozy is too slow on your hardware. |
| LLaMA 7B | Solid general knowledge, neutral tone. Requires stronger prompting to maintain hotel persona. |
| Falcon 7B | Fast but responses feel robotic. Known repetition issues. Not recommended for this use case. |
| Phi-3 Mini (3.8B) | Very fast, very low RAM. Struggles with open-ended concierge queries and persona consistency. |

> **Hardware note:** Snoozy 13B runs at ~120–150s per LLM call on CPU. Since ~70-80% of queries are resolved by FAQ retrieval (sub-second), overall session experience is acceptable for a portfolio demo. For faster inference, swap to Mistral 7B.

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| **UI** | Streamlit |
| **Embeddings** | Sentence-Transformers (MPNet, MiniLM) |
| **Vector Search** | FAISS HNSW |
| **Local LLM** | GPT4All (Snoozy 13B) |
| **Caching** | Session-level multi-layer cache |
| **Metrics** | Custom instrumentation |
| **Core** | Python, NumPy, scikit-learn |

---

## 📂 Project Structure

```
hybrid-rag-concierge/
├── app.py               # Full pipeline — retrieval, routing, LLM, UI, metrics
├── requirements.txt
├── README.md
└── models/              
    ├── all-mpnet-base-v2/
    └── ggml-gpt4all-l13b-snoozy.gguf
```

---

## ⚙️ Installation & Running

1. **Clone the repository**

```bash
git clone https://github.com/SWARNIM-TIWARI/hybrid-rag-concierge.git
cd hybrid-rag-concierge
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download models** and place them in the `models/` folder:

| Model | Link |
|---|---|
| `all-mpnet-base-v2` | [HuggingFace](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) |
| `ggml-gpt4all-l13b-snoozy.gguf` | [GPT4All](https://gpt4all.io) |

5. **Update model paths** in `app.py` to point to your local `models/` folder.

6. **Run**

```bash
streamlit run app.py
```

---

## ⚠️ Explicit Limitations

- Model files are large and not included — must be downloaded separately
- Hard-coded model paths in `app.py` must be updated to match your local setup
- LLM inference on CPU is slow (~120–150s for Snoozy 13B) — this is a hardware constraint, not a code issue
- FAISS index is rebuilt on every session start, not persisted to disk
- Designed for demonstration and learning purposes, not production deployment

---

## 🔭 Future Direction

- Persistent FAISS index saved to disk for faster startup
- Dynamic FAQ loading from JSON or CSV without modifying code
- GPU inference support for faster LLM responses
- Multi-property support with switchable FAQ datasets
- Docker containerization for portable deployment

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

*Built to demonstrate that a well-designed retrieval-first system can outperform cloud-dependent chatbots in privacy-sensitive, latency-critical environments — when the architecture is right, the LLM becomes the exception, not the rule.*