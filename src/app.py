"""Interactive command-line chat for the GDPR Smart Agent.

A simple REPL over the same routed agent used by the API and MCP server. Each
turn shows which retriever was chosen and lists the cited sources.

    python -m src.app

Type 'exit' or 'quit' (or Ctrl-C) to leave.
"""

from __future__ import annotations

from dotenv import load_dotenv

from src.agent.graph import answer_question

load_dotenv()

_EXIT = {"exit", "quit", "q"}


def _print_answer(result: dict) -> None:
    print(f"\n[route: {result.get('route')}]")
    print(result["answer"])
    sources = result.get("chunks", [])
    if sources:
        print("\nSources:")
        for chunk in sources:
            print(f"  [{chunk.citation}] {chunk.title}")
    print()


def main() -> None:
    print("GDPR Smart Agent - ask a question (type 'exit' to quit).\n")
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in _EXIT:
            print("Bye.")
            break

        try:
            result = answer_question(question)
            _print_answer(result)
        except Exception as exc:  # keep the REPL alive on transient errors
            print(f"\n[error] {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
