from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from policy_loader import load_policy_documents

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / ".chroma_index"
COLLECTION_NAME = "university_it_policies"

EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
EMBED_LOCAL_DIR = ROOT / "models" / "bge-small-en-v1.5"


class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_path: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. Install with: "
                "uv pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(model_path, device="cpu")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        vector = self._model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )
        return vector[0].tolist()


def _ensure_embedding_model() -> str:
    env_path = os.environ.get("EMBED_MODEL_PATH")
    if env_path and Path(env_path).exists():
        return env_path
    if (EMBED_LOCAL_DIR / "config.json").exists():
        return str(EMBED_LOCAL_DIR)

    from modelscope import snapshot_download

    print(f"[rag_tool] Downloading embedding model {EMBED_MODEL_ID} from ModelScope ...")
    path = snapshot_download(EMBED_MODEL_ID, local_dir=str(EMBED_LOCAL_DIR))
    print(f"[rag_tool] Embedding model ready at: {path}")
    return str(path)


def _get_chroma_cls():
    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma
    return Chroma


def build_vector_store(force_rebuild: bool = False):
    documents: List[Document] = load_policy_documents()

    if force_rebuild and INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)

    embedding = LocalSentenceTransformerEmbeddings(_ensure_embedding_model())
    Chroma = _get_chroma_cls()

    if INDEX_DIR.exists() and any(INDEX_DIR.iterdir()):
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding,
            persist_directory=str(INDEX_DIR),
        )
        if vector_store._collection.count() >= len(documents):
            print(f"[rag_tool] Reusing existing index "
                  f"({vector_store._collection.count()} policy blocks).")
            return vector_store
        print("[rag_tool] Index incomplete; rebuilding.")
        shutil.rmtree(INDEX_DIR)

    print(f"[rag_tool] Embedding {len(documents)} policy blocks into Chroma ...")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        collection_name=COLLECTION_NAME,
        persist_directory=str(INDEX_DIR),
    )
    print("[rag_tool] Vector store ready.")
    return vector_store


_vector_store = None


def get_vector_store(force_rebuild: bool = False):
    global _vector_store
    if _vector_store is None or force_rebuild:
        _vector_store = build_vector_store(force_rebuild=force_rebuild)
    return _vector_store


def get_retriever(k: int = 4):
    return get_vector_store().as_retriever(search_kwargs={"k": k})


def retrieve(query: str, k: int = 6) -> List[Document]:
    docs = get_retriever(k=max(k * 3, 12)).invoke(query)
    unique, seen = [], set()
    for doc in docs:
        key = " ".join(doc.page_content.lower().split())
        if key not in seen:
            seen.add(key)
            unique.append(doc)
        if len(unique) >= k:
            break
    return unique


def format_context(documents: List[Document]) -> str:
    if not documents:
        return "(No relevant policy excerpts were found.)"
    blocks = []
    for doc in documents:
        blocks.append(
            f"[Excerpt {doc.metadata.get('policy_id', '?')} | "
            f"topic: {doc.metadata.get('topic', '?')}]\n{doc.page_content}"
        )
    return "\n\n".join(blocks)


def retrieve_context(query: str, k: int = 4) -> Tuple[str, List[Document]]:
    docs = retrieve(query, k=k)
    return format_context(docs), docs


def build_rag_tool(k: int = 4):
    from langchain_core.tools import tool

    @tool(
        "policy_search",
        return_direct=False,
        description=(
            "Search the university IT policy knowledge base for supported facts, "
            "including contact details, VPN, Wi-Fi, printing, accounts, email "
            "and storage."
        ),
    )
    def policy_search(query: str) -> str:
        context, _ = retrieve_context(query, k=k)
        return context

    return policy_search


if __name__ == "__main__":
    import sys

    question = sys.argv[1] if len(sys.argv) > 1 else (
        "What exact phone number should I call for the University IT Service Desk?"
    )
    context, docs = retrieve_context(question)
    print(f"QUESTION: {question}\n")
    print(f"Retrieved {len(docs)} document(s):")
    for d in docs:
        print("  -", d.metadata.get("policy_id"), "|", d.metadata.get("topic"))
    print("\nCONTEXT:\n" + context)
