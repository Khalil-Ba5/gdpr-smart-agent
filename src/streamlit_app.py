"""Streamlit chat frontend for the GDPR Smart Agent.

A thin client over the FastAPI ``/ask`` endpoint (src/api.py) — no agent logic
lives here, same principle as the CLI (src/app.py) and MCP server. Run the API
first, then this:

    uvicorn src.api:app --reload
    streamlit run src/streamlit_app.py

Requires `requests` and `streamlit` (both in requirements.txt).
"""

from __future__ import annotations

import requests
import streamlit as st

DEFAULT_API_BASE = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = 60  # seconds; LLM calls can be slow

st.set_page_config(page_title="GDPR Smart Agent", page_icon="⚖️", layout="centered")


def _init_state() -> None:
    st.session_state.setdefault("messages", [])  # list[{role, content, sources, route}]
    st.session_state.setdefault("api_base", DEFAULT_API_BASE)


def _check_health(api_base: str) -> bool:
    try:
        resp = requests.get(f"{api_base}/health", timeout=5)
        return resp.ok
    except requests.RequestException:
        return False


def _ask(api_base: str, question: str, k: int) -> dict:
    resp = requests.post(
        f"{api_base}/ask",
        json={"question": question, "k": k},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for src in sources:
            title = f" — {src['title']}" if src.get("title") else ""
            st.markdown(f"**[{src['citation']}]**{title}")
            if src.get("url"):
                st.caption(src["url"])


def _render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        if msg.get("route"):
            st.caption(f"routed to: {msg['route']}")
        st.markdown(msg["content"])
        if msg.get("sources"):
            _render_sources(msg["sources"])


def main() -> None:
    _init_state()

    with st.sidebar:
        st.header("Settings")
        st.session_state.api_base = st.text_input("API base URL", st.session_state.api_base)
        k = st.slider("Passages to retrieve (k)", min_value=1, max_value=20, value=4)

        if _check_health(st.session_state.api_base):
            st.success("API reachable")
        else:
            st.error("API unreachable — start it with:\n\n`uvicorn src.api:app --reload`")

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    st.title("⚖️ GDPR Smart Agent")
    st.caption("Ask a question about the EU GDPR — answers are grounded and cited.")

    for msg in st.session_state.messages:
        _render_message(msg)

    question = st.chat_input("Ask a GDPR question…")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Thinking…")
        try:
            result = _ask(st.session_state.api_base, question, k)
        except requests.RequestException as exc:
            placeholder.error(f"Request failed: {exc}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Request failed: {exc}"}
            )
            return

        placeholder.empty()
        route = result.get("route")
        if route:
            st.caption(f"routed to: {route}")
        st.markdown(result["answer"])
        sources = result.get("sources", [])
        _render_sources(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": sources,
            "route": route,
        }
    )


if __name__ == "__main__":
    main()
