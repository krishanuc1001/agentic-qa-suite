"""
Observability bootstrap (Gap #9).

This module is intentionally tiny and dependency-free so it can be imported
unconditionally at app start. It does three things:

1. Configures stdlib logging in a structured single-line format that is
   friendly to log aggregators (Streamlit Cloud, HF Spaces, Render, Fly).
2. Auto-enables LangSmith tracing when `LANGCHAIN_API_KEY` (or
   `LANGSMITH_API_KEY`) is present in the environment / Streamlit secrets,
   so a deployer gets traces by simply setting a secret -- no code change.
3. Exposes `get_token_counter_callback()` which returns a LangChain callback
   that aggregates per-run prompt / completion / total token usage. Hook
   it into ``llm.invoke(..., config={"callbacks": [cb]})`` if you want a
   per-run cost meter (Gap #4 cost guardrail).

Idempotent: safe to call ``setup_observability()`` multiple times.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

_DONE = False

_LOG_FORMAT = (
    "%(asctime)s level=%(levelname)s logger=%(name)s "
    "msg=%(message)s"
)


def setup_observability() -> None:
    """Configure logging and (optionally) LangSmith tracing exactly once."""
    global _DONE
    if _DONE:
        return

    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=log_level, format=_LOG_FORMAT, force=False)

    # Quiet down noisy third-party loggers in interactive demo mode.
    for noisy in ("httpx", "httpcore", "urllib3", "openai", "google"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # ---- LangSmith auto-wire (no code changes required to enable) ----
    # Accept either langchain-style or langsmith-style env names.
    api_key = os.environ.get("LANGCHAIN_API_KEY") or os.environ.get("LANGSMITH_API_KEY")
    if api_key and not os.environ.get("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_API_KEY"] = api_key
    if api_key:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", "qa-intelligence-suite")
        logging.getLogger(__name__).info(
            "LangSmith tracing enabled (project=%s).",
            os.environ["LANGCHAIN_PROJECT"],
        )

    _DONE = True


def get_token_counter_callback() -> Optional[Any]:
    """
    Returns a callback handler that aggregates token usage across LLM calls
    in a single run. Returns ``None`` if the optional callback class is not
    importable (older langchain-core); callers should treat that as a no-op.

    Usage:

        cb = get_token_counter_callback()
        if cb is not None:
            llm.invoke(messages, config={"callbacks": [cb]})
        # cb.totals -> {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}
    """
    try:
        from langchain_core.callbacks import BaseCallbackHandler
    except Exception:
        return None

    class _TokenCounter(BaseCallbackHandler):
        def __init__(self) -> None:
            super().__init__()
            self.totals: Dict[str, int] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

        def on_llm_end(self, response, **kwargs):  # type: ignore[override]
            try:
                gens = response.generations or []
                for batch in gens:
                    for gen in batch:
                        msg = getattr(gen, "message", None)
                        usage = (
                            getattr(msg, "usage_metadata", None)
                            or (response.llm_output or {}).get("token_usage")
                            or {}
                        )
                        if not usage:
                            continue
                        self.totals["prompt_tokens"] += int(
                            usage.get("input_tokens") or usage.get("prompt_tokens") or 0
                        )
                        self.totals["completion_tokens"] += int(
                            usage.get("output_tokens") or usage.get("completion_tokens") or 0
                        )
                        self.totals["total_tokens"] += int(
                            usage.get("total_tokens")
                            or (
                                (usage.get("input_tokens") or 0)
                                + (usage.get("output_tokens") or 0)
                            )
                        )
            except Exception:
                # Never let a metering bug break a real run.
                pass

    return _TokenCounter()
