"""Grounded RAG agent for the university IT policy exercise.

Pipeline (deterministic retrieve-then-read, appropriate for a 0.6B model):
    question -> rag_tool.policy_search (Chroma retrieval over the single policy
    file) -> a strictly grounded Qwen3-0.6B prompt -> greedy generation.

Hard requirement of the exercise: for "What exact phone number should I call
for the University IT Service Desk?" the agent must say it does not know / the
number is not specified, and must never invent one. The knowledge base contains
no phone number, so retrieval cannot surface one; the prompt instructs
abstention, and a small groundedness guard replaces any telephone number that
is not supported by the retrieved excerpts.

The LLM is the LOCAL Qwen3-0.6B model (NOT the 8B / Ollama model from the
lecture example).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional

import torch
from langchain_core.language_models.llms import LLM
from langchain_core.runnables import RunnableLambda

import rag_tool
from rag_tool import retrieve_context

ROOT = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen3-0.6B"

# Fixed abstention wording used whenever the answer is not in the excerpts.
ABSTAIN = (
    "I'm sorry, but I don't have that information; the university IT "
    "guidance I was given does not specify it."
)

# Matches common telephone spellings, e.g. 0800 123 4567, +358 40 1234567,
# (555) 123-4567, 555-123-4567.
PHONE_RE = re.compile(r"(\+?\d[\d\s().\-]{6,}\d)")

PROMPT_TEMPLATE = (
    "You are a university IT support assistant. You answer strictly from the "
    "Policy excerpts provided.\n"
    "Rules:\n"
    "1. If the excerpts state the answer, give it directly in one or two short "
    "sentences, using only those facts. Times and numbers that ARE in the "
    "excerpts (for example 15 minutes or 24 hours) must be used in your answer.\n"
    "2. If the specific requested detail is NOT stated in the excerpts (for "
    "example a telephone number that the excerpts do not list), reply only that "
    "you do not have that information because the provided guidance does not "
    "specify it. Never invent or guess a phone number, room number or any other "
    "missing detail.\n\n"
    "Example 1\n"
    "Policy excerpts:\n"
    "Students release a print job at a printer by authenticating with their "
    "student card.\n"
    "Question: How do I release a queued print job at the printer?\n"
    "Answer: Authenticate at the printer with your student card to release the job.\n\n"
    "Example 2\n"
    "Policy excerpts:\n"
    "A temporarily locked account should be left for 15 minutes before another "
    "sign-in attempt. If an account remains locked, contact the IT Service Desk.\n"
    "Question: My account locked six minutes ago after several wrong passwords. "
    "Should I keep trying to sign in?\n"
    "Answer: No. Wait 15 minutes before trying to sign in again; if it is still "
    "locked after that, contact the IT Service Desk.\n\n"
    "Example 3\n"
    "Policy excerpts:\n"
    "If an account remains locked, the user should contact the IT Service Desk.\n"
    "Question: What is the exact phone number of the IT Service Desk?\n"
    "Answer: " + ABSTAIN + "\n\n"
    "Now answer the following using only its policy excerpts.\n\n"
    "Policy excerpts:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

_STOP_MARKERS = (
    "\nQuestion:", "\nPolicy excerpts:", "\nUser:", "\nAnswer:", "\nAnswer",
    "\n---", "\n\n\n", "\n",
)  # the final "\n" keeps only the first answer line (answers are 1-2 sentences)


def _truncate_repetition(text: str, ngram: int = 4, max_sentences: int = 4) -> str:
    """Trim a greedy base model's degenerate phrase loops.

    Keeps the correct prefix, then cuts at the first repeated word n-gram
    (rounded back to the preceding sentence boundary), and caps sentence count.
    """
    words = text.split()
    cut = len(words)
    seen = set()
    for i in range(ngram, len(words) + 1):
        gram = tuple(w.strip(".,;:!?\"'()[]").lower() for w in words[i - ngram:i])
        if gram in seen:
            cut = i - ngram
            break
        seen.add(gram)
    text = " ".join(words[:cut]).strip()
    # Drop a trailing clause fragment (one without a sentence-ending mark).
    ends = list(re.finditer(r"[.!?]", text))
    if ends:
        text = text[: ends[-1].end()].strip()
    # Cap the number of complete sentences.
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]
    return " ".join(sentences).strip()


# --------------------------------------------------------------------------- #
# Local Qwen3-0.6B model resolution and LangChain LLM wrapper
# --------------------------------------------------------------------------- #
def resolve_model_path() -> str:
    """Locate a local Qwen3-0.6B directory, otherwise fetch it from ModelScope."""
    env = os.environ.get("QWEN3_MODEL_PATH")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates += [
        ROOT / "models" / "qwen3-0.6b",
        ROOT.parent / "SCSE_26_GEX3" / "models" / "qwen3-0.6b",
        ROOT.parent / "SCSE26_GEX3" / "models" / "qwen3-0.6b",
    ]
    for path in candidates:
        if path is not None and (path / "config.json").exists():
            return str(path)

    # Fall back to a ModelScope download inside this repository (VPN off).
    from modelscope import snapshot_download

    target = str(ROOT / "models" / "qwen3-0.6b")
    print(f"[agent] Local Qwen3-0.6B not found; downloading {MODEL_ID} from ModelScope ...")
    return snapshot_download(MODEL_ID, local_dir=target)


class QwenLocalLLM(LLM):
    """A minimal LangChain LLM that runs Qwen3-0.6B locally on CPU."""

    model_path: str
    max_new_tokens: int = 96

    def __init__(self, model_path: Optional[str] = None, **kwargs):
        super().__init__(model_path=model_path or resolve_model_path(), **kwargs)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(2)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path, dtype="auto"
        )
        self._model.eval()

    @property
    def _llm_type(self) -> str:
        return "qwen3-0.6b-local"

    def _generate_text(self, prompt: str) -> str:
        inputs = self._tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            # Plain greedy decoding gives the most faithful extraction on this
            # small model; any degenerate repetition is trimmed afterwards.
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        for marker in _STOP_MARKERS:
            text = text.split(marker)[0]
        return _truncate_repetition(text.strip())

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        return self._generate_text(prompt)


# --------------------------------------------------------------------------- #
# Groundedness guard
# --------------------------------------------------------------------------- #
def _contains_phone(text: str) -> bool:
    for match in PHONE_RE.findall(text):
        digits = re.sub(r"\D", "", match)
        if len(digits) >= 7:  # a plausible telephone number
            return True
    return False


def _grounded_answer(raw_answer: str, context: str) -> str:
    """Reject answers that state a phone number absent from the excerpts."""
    if _contains_phone(raw_answer) and not _contains_phone(context):
        return ABSTAIN
    answer = raw_answer.strip()
    if not answer:
        return ABSTAIN
    return answer


# --------------------------------------------------------------------------- #
# The RAG agent
# --------------------------------------------------------------------------- #
class RAGAgent:
    """Retrieves policy excerpts, then answers strictly from them with Qwen3-0.6B."""

    def __init__(self, k: int = 6, model_path: Optional[str] = None,
                 max_new_tokens: int = 96):
        self.k = k
        self.llm = QwenLocalLLM(model_path=model_path, max_new_tokens=max_new_tokens)
        # Retrieval exposed as a LangChain tool (the tool the agent "calls").
        self.search_tool = rag_tool.build_rag_tool(k=k)

    def retrieve(self, question: str):
        return retrieve_context(question, k=self.k)

    def answer(self, question: str, return_sources: bool = False):
        context, docs = self.retrieve(question)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        raw = self.llm._generate_text(prompt)
        final = _grounded_answer(raw, context)
        if return_sources:
            sources = [d.metadata.get("policy_id") for d in docs]
            return final, sources, context, raw
        return final

    # LangChain Runnable interface: build_agent() | agent.invoke({"question": ...})
    def invoke(self, input_, *args, **kwargs):
        question = input_["question"] if isinstance(input_, dict) else input_
        return self.answer(question)


_agent: Optional[RAGAgent] = None


def build_agent(k: int = 6, **kwargs) -> RAGAgent:
    global _agent
    if _agent is None:
        _agent = RAGAgent(k=k, **kwargs)
    return _agent


def answer_question(question: str, k: int = 6) -> str:
    """Convenience one-shot entry point used by rag.py."""
    return build_agent(k=k).answer(question)


def as_runnable(k: int = 6):
    """Expose the agent as a LangChain RunnableLambda (question -> answer)."""
    return RunnableLambda(
        lambda inp: build_agent(k=k).answer(
            inp["question"] if isinstance(inp, dict) else inp
        )
    )


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else (
        "What exact phone number should I call for the University IT Service Desk?"
    )
    agent = build_agent()
    answer, sources, context, raw = agent.answer(q, return_sources=True)
    print("QUESTION:", q)
    print("SOURCES:", sources)
    print("RAW MODEL:", repr(raw))
    print("FINAL ANSWER:", answer)
