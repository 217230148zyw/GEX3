import sys

import agent
import rag_tool

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
