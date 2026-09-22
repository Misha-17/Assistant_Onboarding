"""Run the actual assistant on an isolated synthetic onboarding handbook."""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

from .config import Config
from .engine import SisuReader


HANDBOOK = """# Section 1 Remote work

Employees may work remotely, subject to Section 2.

# Section 2 Eligibility

The remote-work permission is limited by the onboarding exception in Section 3.

# Section 3 Onboarding exception

During their first month, new employees require manager approval before working remotely.

# Section 4 Equipment

New employees may request a company laptop, subject to Section 8.

# Section 5 Help

The IT service desk handles laptop requests.
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4:12b")
    parser.add_argument("--question", default="Can a new employee work remotely without manager approval?")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="sisu-reference-demo-") as temp:
        root = Path(temp)
        source = root / "handbook.md"
        source.write_text(HANDBOOK, encoding="utf-8")
        config = Config(project_dir=root, workspace_dir=root / "workspace", model=args.model, trace_mode="off")
        engine = SisuReader(config)
        try:
            engine.rebuild((source,))
            answer = engine.ask(args.question)
            print(f"Status: {answer.status}\n{answer.text}")
            for warning in answer.warnings:
                print(f"Note: {warning}")
            for citation in answer.sources:
                print(f"[{citation.source_id}] {citation.locator}: {citation.quote}")
        finally:
            engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
