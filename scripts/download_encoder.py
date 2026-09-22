"""Install the pinned E5 artifacts, or copy a verified existing cache.

Uses only Python's standard library. No remote Python code is imported.
Downloads resume through .part files and become final only after SHA-256 checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "model_manifests/multilingual-e5-small.json"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root, name):
    """Reject path traversal and Windows drive/ADS paths on every platform."""
    relative = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or relative.is_absolute() or
            any(part in {"..", ".", ""} for part in name.split("/"))):
        raise ValueError("Unsafe artifact path: " + name)
    target = (Path(root) / name).resolve()
    if not target.is_relative_to(Path(root).resolve()):
        raise ValueError("Artifact escapes destination")
    return target


def matches(path, item):
    return (path.is_file() and path.stat().st_size == item["bytes"] and
            sha256(path) == item["sha256"])


class HttpsRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        if urlsplit(newurl).scheme != "https":
            raise ValueError("Refusing a non-HTTPS redirect")
        return super().redirect_request(request, fp, code, message, headers, newurl)


def fetch(item, target, *, opener=None):
    if urlsplit(item["url"]).scheme != "https":
        raise ValueError("Artifacts require HTTPS")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    if partial.is_symlink():
        raise ValueError("Refusing a symlink partial download")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset >= item["bytes"]:
        if matches(partial, item):
            partial.replace(target)
            return
        offset = 0
    headers = {"User-Agent": "SISU-pinned-model-installer/1", "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(item["url"], headers=headers)
    opener = opener or urllib.request.build_opener(HttpsRedirect())
    with opener.open(request, timeout=90) as response:
        status = response.status
        if offset and status == 206:
            value = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value)
            if (not match or int(match[1]) != offset or
                    int(match[3]) != item["bytes"] or int(match[2]) != item["bytes"] - 1):
                raise ValueError("Unexpected partial-download range")
            mode = "ab"
        elif status == 200:
            offset, mode = 0, "wb"
        else:
            raise ValueError("Unexpected download response")
        with partial.open(mode) as stream:
            while True:
                chunk = response.read(min(1024 * 1024, item["bytes"] - offset + 1))
                if not chunk:
                    break
                offset += len(chunk)
                if offset > item["bytes"]:
                    raise ValueError("Download exceeds pinned size")
                stream.write(chunk)
    if not matches(partial, item):
        raise ValueError("Downloaded artifact failed its size/SHA-256 check")
    partial.replace(target)


def install_files(items, destination, *, source=None, verify_only=False, opener=None):
    seen = set()
    for item in items:
        name = item["path"]
        if (name in seen or type(item["bytes"]) is not int or item["bytes"] < 0 or
                not re.fullmatch(r"[a-f0-9]{64}", item["sha256"])):
            raise ValueError("Invalid or duplicate manifest artifact")
        seen.add(name)
        safe_path(destination, name)
        if source is not None:
            safe_path(source, name)
    for number, item in enumerate(items, 1):
        target = safe_path(destination, item["path"])
        print(f"[{number}/{len(items)}] {item['path']}", flush=True)
        if matches(target, item):
            print("  verified existing file", flush=True)
            continue
        if target.exists():
            raise ValueError(f"Existing file differs from the pin; move it aside before retrying: {target}")
        if verify_only:
            raise FileNotFoundError("Missing pinned artifact: " + str(target))
        if source is not None:
            original = safe_path(source, item["path"])
            if not matches(original, item):
                raise ValueError("Offline source differs from pin: " + str(original))
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            if partial.is_symlink():
                raise ValueError("Refusing a symlink partial file")
            shutil.copyfile(original, partial)
            if not matches(partial, item):
                raise ValueError("Offline copy failed verification")
            partial.replace(target)
        else:
            for attempt in range(3):
                try:
                    fetch(item, target, opener=opener)
                    break
                except (OSError, urllib.error.URLError):
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
        print("  installed and verified", flush=True)


def write_manifest(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == raw:
        return
    temporary = path.with_name(path.name + ".tmp")
    if temporary.is_symlink() or path.is_symlink():
        raise ValueError("Refusing a symlink manifest")
    temporary.write_bytes(raw)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path,
                        default=Path(os.environ.get("SISU_READER_ENCODER_PATH") or ROOT / "models/multilingual-e5-small"))
    parser.add_argument("--source", type=Path, help="Copy these exact files from an existing local E5 directory; no network")
    parser.add_argument("--verify-only", action="store_true", help="Verify all existing files without downloading or writing")
    args = parser.parse_args()
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    if (manifest["schema"] != "sisu.pinned_encoder.v1" or
            manifest["repository"] != "intfloat/multilingual-e5-small" or manifest["revision"] != REVISION):
        raise ValueError("Unexpected encoder declaration")
    prefix = f"https://huggingface.co/{manifest['repository']}/resolve/{REVISION}/"
    if any(item["url"] != prefix + item["path"] for item in manifest["files"]):
        raise ValueError("Artifact URL does not match the declared revision")
    install_files(manifest["files"], args.destination, source=args.source, verify_only=args.verify_only)
    if not args.verify_only:
        write_manifest(args.destination / "manifest.json", raw)
    else:
        if (args.destination / "manifest.json").read_bytes() != raw:
            raise ValueError("Installed manifest differs from bundled declaration")
    print("Pinned encoder ready at " + str(args.destination.resolve()))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        print("Encoder setup failed: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
