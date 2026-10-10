"""Entry point for the university IT policy RAG system.

Usage:
    python rag.py
        -> asks the graded question about the IT Service Desk phone number.
    python rag.py "Your question here"
        -> answers an arbitrary question from the policy knowledge base.
    python rag.py --show-context "Your question"
        -> also prints the retrieved policy excerpts.

The model is the local Qwen3-0.6B. If the knowledge base does not contain the
answer (it contains no Service Desk phone number), the agent must state that it
does not know rather than inventing a number.
"""

import sys

import agent
import rag_tool  # noqa: F401  (importing builds/validates the retrieval stack)

DEFAULT_QUESTION = (
    "What exact phone number should I call for the University IT Service Desk?"
)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    show_context = False
    if "--show-context" in argv:
        show_context = True
        argv.remove("--show-context")

    question = " ".join(argv).strip() if argv else DEFAULT_QUESTION

    print("=" * 60)
    print("RAG question:", question)
    print("=" * 60)

    rag_agent = agent.build_agent()
    answer, sources, context, raw = rag_agent.answer(question, return_sources=True)

    print("\nRetrieved policy excerpts:", ", ".join(sources) if sources else "(none)")
    if show_context:
        print("\n----- RETRIEVED CONTEXT -----")
        print(context)
        print("-----------------------------")
    print("\nRaw model output:", repr(raw))
    print("\nFINAL ANSWER:")
    print(answer)
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
