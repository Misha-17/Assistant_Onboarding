"""Install the optional pinned MiniCheck model and reviewed upstream methods.

No model or remote Python code is executed by this setup helper. The application
later verifies the upstream hash before extracting three reviewed scoring methods.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from download_encoder import ROOT, install_files, write_manifest

MANIFEST = ROOT / "model_manifests/minicheck-flan-t5-large.json"
MODEL_REVISION = "96eafd01cee2d16cf81aaa2fb226b14f422a37b3"
CODE_REVISION = "b58b9fa69acbd1015ec970fa65dd752413a053d2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "models/minicheck")
    parser.add_argument("--source", type=Path,
                        help="Existing portable MiniCheck cache with model/ and upstream--COMMIT/; no network")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if (data["repository"] != "lytang/MiniCheck-Flan-T5-Large" or
            data["revision"] != MODEL_REVISION or data["upstream_commit"] != CODE_REVISION or
            data["local_path"] != "model"):
        raise ValueError("Unexpected MiniCheck declaration")
    files = [{"path": "model/" + item["name"], "sha256": item["sha256"], "bytes": item["bytes"],
              "url": f"https://huggingface.co/{data['repository']}/resolve/{MODEL_REVISION}/{item['name']}"}
             for item in data["files"]]
    files += [{"path": f"upstream--{CODE_REVISION}/" + item["path"], "sha256": item["sha256"],
               "bytes": item["bytes"],
               "url": f"https://raw.githubusercontent.com/Liyan06/MiniCheck/{CODE_REVISION}/{item['path']}"}
              for item in data["upstream_files"]]
    install_files(files, args.destination, source=args.source, verify_only=args.verify_only)
    raw = (json.dumps(data, indent=2) + "\n").encode("utf-8")
    if args.verify_only:
        if (args.destination / "manifest.json").read_bytes() != raw:
            raise ValueError("Installed manifest differs from bundled declaration")
    else:
        write_manifest(args.destination / "manifest.json", raw)
    print("Optional MiniCheck assets ready at " + str(args.destination.resolve()))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        print("MiniCheck setup failed: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
