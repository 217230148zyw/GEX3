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

ABSTAIN = (
    "I'm sorry, but I don't have that information; the university IT "
    "guidance I was given does not specify it."
)

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
)


def _truncate_repetition(text: str, ngram: int = 4, max_sentences: int = 4) -> str:
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
    ends = list(re.finditer(r"[.!?]", text))
    if ends:
        text = text[: ends[-1].end()].strip()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]
    return " ".join(sentences).strip()


def resolve_model_path() -> str:
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

    from modelscope import snapshot_download

    target = str(ROOT / "models" / "qwen3-0.6b")
    print(f"[agent] Local Qwen3-0.6B not found; downloading {MODEL_ID} from ModelScope ...")
    return snapshot_download(MODEL_ID, local_dir=target)


class QwenLocalLLM(LLM):
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


def _contains_phone(text: str) -> bool:
    for match in PHONE_RE.findall(text):
        digits = re.sub(r"\D", "", match)
        if len(digits) >= 7:
            return True
    return False


def _grounded_answer(raw_answer: str, context: str) -> str:
    if _contains_phone(raw_answer) and not _contains_phone(context):
        return ABSTAIN
    answer = raw_answer.strip()
    if not answer:
        return ABSTAIN
    return answer


class RAGAgent:
    def __init__(self, k: int = 6, model_path: Optional[str] = None,
                 max_new_tokens: int = 96):
        self.k = k
        self.llm = QwenLocalLLM(model_path=model_path, max_new_tokens=max_new_tokens)
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
    return build_agent(k=k).answer(question)


def as_runnable(k: int = 6):
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
    active_agent = build_agent()
    answer, sources, context, raw = active_agent.answer(q, return_sources=True)
    print("QUESTION:", q)
    print("SOURCES:", sources)
    print("RAW MODEL:", repr(raw))
    print("FINAL ANSWER:", answer)
