"""Run the standalone V10 assistant with the validated product defaults."""
from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def configure_environment() -> None:
    defaults = {
        "SISU_READER_WORKSPACE": str(ROOT / "workspace"),
        "SISU_READER_MODEL": "gpt-oss:20b",
        "SISU_READER_CONTEXT_TOKENS": "32768",
        "SISU_READER_DEADLINE_S": "95",
        "SISU_READER_MAX_MODEL_CALLS": "30",
        "SISU_READER_MAX_OUTPUT_TOKENS": "32768",
        "SISU_READER_WEB_PORT": "8782",
        "SISU_READER_TRACE_MODE": "answer",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)
    # V9 controller actions cannot be applied to V10's different pipeline.
    os.environ["SISU_READER_STRATEGY_LEARNING"] = "false"


def main(argv: list[str] | None = None) -> int:
    configure_environment()
    from assistant_runtime.__main__ import main as assistant_main
    return assistant_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
