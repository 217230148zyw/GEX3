# SCSE '26 — Graded Exercise #3 (RAG part)

Retrieval-Augmented Generation over a **single** university IT policy file,
using the **local Qwen3-0.6B** model (not the 8B/Ollama model from Lecture 11).

## Files

| File | Role |
| --- | --- |
| `policy_loader.py` | **Fixed** loader for `university_it_policies.txt` — not modified. |
| `rag_tool.py` | **Retrieval ("R")**: loads the documents with `load_policy_documents()`, embeds them with a small local `bge-small-en-v1.5` SentenceTransformer and stores/searches them in a persistent Chroma vector store; also exposes retrieval as a LangChain tool (`policy_search`). |
| `agent.py` | **Agent + generation ("AG")**: resolves and loads local Qwen3-0.6B, retrieves policy excerpts, builds a strictly grounded prompt and answers greedily; abstains when the fact is absent and refuses to invent phone numbers (groundedness guard). |
| `rag.py` | **Entry point / CLI**. |

## Setup

```bash
python -m venv .venv          # or: uv venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt   # or: pip install -r requirements.txt
```

Qwen3-0.6B is found automatically in this order:
`$QWEN3_MODEL_PATH` → `./models/qwen3-0.6b` → `../SCSE_26_GEX3/models/qwen3-0.6b`
→ otherwise downloaded from ModelScope into `./models/qwen3-0.6b` (keep VPN off).
The embedding model is likewise fetched from ModelScope on first run.

## Run

```bash
# Graded question (default):
python rag.py

# Any other question:
python rag.py "How long are unreleased print jobs kept?"

# Show the retrieved excerpts:
python rag.py --show-context "..."
```

## Expected behaviour

The knowledge base (`university_it_policies.txt`, 1,196 policy blocks) contains
**no verified IT Service Desk telephone number**. Asking

> *What exact phone number should I call for the University IT Service Desk?*

therefore returns an answer such as

> *I'm sorry, but I don't have that information; the university IT guidance I
> was given does not specify it.*

The model does **not** hallucinate a number and does not return an irrelevant
answer. `python check_data.py` confirms the source dataset supplies no phone
number.
