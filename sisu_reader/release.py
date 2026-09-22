"""Build and validate a deterministic, source-only SISU Reader archive.

The release archive is assembled from an explicit allowlist. Runtime
workspaces, databases, traces, locks, caches, credentials, and personal home
paths are never valid release inputs. The builder uses only the standard
library and does not load the research engine or contact a model.

No software license is selected here. If the project has no LICENSE file, the
archive is suitable only for internal evaluation until the owner or legal team
defines redistribution terms.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


RELEASE_FORMAT_VERSION = 1
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
_MAX_FILE_BYTES = 16 * 1024 * 1024
_MAX_TOTAL_BYTES = 64 * 1024 * 1024
_MAX_OFFICE_EXPANDED_BYTES = 32 * 1024 * 1024

_ROOT_ALLOWLIST = frozenset(
    {
        ".gitignore",
        "ARCHITECTURE.md",
        "AUDIT.md",
        "BACKLOG.md",
        "EVALUATION.md",
        "IMPLEMENTATION_REPORT.md",
        "LICENSE",
        "LICENSE.md",
        "LICENSE.txt",
        "MIGRATION.md",
        "NOTICE",
        "PRIORITY_PLAN.md",
        "README.md",
        "ROBUSTNESS.md",
        "RELEASING.md",
        "SECURITY.md",
        "TECHNICAL_GUIDE.md",
        "pyproject.toml",
        "requirements.lock",
        "sisu-reader",
        "sisu-reader.cmd",
    }
)

# These are the task's required reports plus the minimum runnable source and
# release controls. License is intentionally not required because selecting
# redistribution terms is a legal/ownership decision, not an engineering one.
REQUIRED_RELEASE_PATHS = frozenset(
    {
        ".github/workflows/ci.yml",
        ".gitignore",
        "ARCHITECTURE.md",
        "AUDIT.md",
        "BACKLOG.md",
        "EVALUATION.md",
        "IMPLEMENTATION_REPORT.md",
        "PRIORITY_PLAN.md",
        "README.md",
        "SECURITY.md",
        "TECHNICAL_GUIDE.md",
        "pyproject.toml",
        "requirements.lock",
        "sisu_reader/__init__.py",
        "sisu_reader/release.py",
        "tests/test_release.py",
    }
)

_PACKAGE_SUFFIXES = frozenset(
    {".py", ".html", ".css", ".js", ".json", ".sql", ".svg"}
)
_TEST_SUFFIXES = frozenset(
    {
        ".py",
        ".json",
        ".md",
        ".txt",
        ".html",
        ".htm",
        ".css",
        ".js",
        ".csv",
        ".tsv",
        ".yml",
        ".yaml",
        ".docx",
        ".pdf",
    }
)
_TEXT_SUFFIXES = frozenset(
    {
        ".bib",
        "",
        ".cmd",
        ".css",
        ".csv",
        ".html",
        ".htm",
        ".ini",
        ".js",
        ".json",
        ".lock",
        ".md",
        ".py",
        ".sql",
        ".svg",
        ".toml",
        ".tsv",
        ".txt",
        ".yaml",
        ".yml",
    }
)
_OFFICE_TEXT_SUFFIXES = frozenset({".xml", ".rels", ".txt", ".json"})
_BINARY_FIXTURE_SUFFIXES = frozenset({".docx", ".pdf"})

_DENIED_COMPONENTS = frozenset(
    {
        ".git",
        ".idea",
        ".mypy_cache",
        ".pytest_cache",
        ".pyright",
        ".ruff_cache",
        ".venv",
        ".vscode",
        "__pycache__",
        "build",
        "dist",
        "env",
        "htmlcov",
        "trace",
        "traces",
        "venv",
        "workspace",
    }
)
_DENIED_ENDINGS = (
    ".db",
    ".db-shm",
    ".db-wal",
    ".key",
    ".p12",
    ".pem",
    ".pfx",
    ".pyc",
    ".pyo",
    ".sqlite",
    ".sqlite3",
    ".sqlite3-shm",
    ".sqlite3-wal",
    ".tmp",
)
_DENIED_FILENAMES = frozenset(
    {
        ".env",
        ".reader-rebuild.lock",
        "credentials.json",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "inference.lock",
        "secrets.json",
    }
)

_WINDOWS_PERSONAL_PATH = re.compile(
    r"(?i)(?<![A-Za-z0-9_])[A-Z]:[\\/](?:Users|Documents and Settings)"
    r"[\\/][^\\/\r\n\"'<>|]{1,80}(?:[\\/]|$)"
)
_POSIX_PERSONAL_PATH = re.compile(
    r"(?i)(?<![A-Za-z0-9_])/(?:home|Users)/[A-Za-z0-9._-]{1,80}(?:/|$)"
)
_CREDENTIAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private-key material",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    (
        "GitHub token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    ),
    ("OpenAI-style token", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
    (
        "assigned secret",
        re.compile(
            r"(?i)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)"
            r"\s*[:=]\s*[\"']?[A-Za-z0-9_./+=-]{16,}"
        ),
    ),
)


class ReleaseValidationError(ValueError):
    """Raised when a source tree or archive violates the release contract."""


@dataclass(frozen=True, slots=True)
class ReleaseFile:
    relative_path: str
    data: bytes
    mode: int = 0o644


@dataclass(frozen=True, slots=True)
class ReleasePlan:
    files: tuple[ReleaseFile, ...]
    total_bytes: int
    license_present: bool


@dataclass(frozen=True, slots=True)
class ArchiveReport:
    archive_path: Path
    file_count: int
    total_bytes: int
    sha256: str
    license_present: bool


def _path_parts(relative_path: str) -> tuple[str, ...]:
    return tuple(part.casefold() for part in PurePosixPath(relative_path).parts)


def _denied_path_reason(relative_path: str) -> str:
    if not relative_path or "\x00" in relative_path:
        return "empty or invalid archive path"
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts:
        return "absolute or parent-traversing archive path"
    parts = _path_parts(relative_path)
    if any(part in _DENIED_COMPONENTS for part in parts):
        return "runtime, cache, build, or editor directory"
    if any(part.endswith(".egg-info") for part in parts):
        return "generated package metadata"
    name = parts[-1]
    if name in _DENIED_FILENAMES or (name.startswith(".env.") and name != ".env.example"):
        return "credential or runtime-state filename"
    if name.endswith(".lock") and name != "requirements.lock":
        return "runtime lock file"
    if any(name.endswith(ending) for ending in _DENIED_ENDINGS):
        return "runtime, compiled, database, or credential file"
    return ""


def _allowed_release_path(relative_path: str) -> bool:
    pure = PurePosixPath(relative_path)
    parts = pure.parts
    if len(parts) == 1:
        return relative_path in _ROOT_ALLOWLIST
    suffix = pure.suffix.casefold()
    if parts[:2] == (".github", "workflows"):
        return len(parts) == 3 and suffix in {".yml", ".yaml"}
    if parts[0] == "sisu_reader":
        return suffix in _PACKAGE_SUFFIXES
    if parts[0] == "tests":
        return suffix in _TEST_SUFFIXES
    if parts[:3] == ("research", "baseline", "sisu_reader"):
        return len(parts) == 4 and suffix == ".py"
    if relative_path in {
        "research/ROBUSTNESS_RESEARCH.md", "research/REFERENCES.bib",
        "research/run_live_comparison.py",
        "research/baseline/BASELINE.md",
        "research_results/reference_closure/reference_benchmark.md",
        "research_results/reference_closure/reference_benchmark.json",
        "research_results/reference_closure/reference_fixtures.json",
        "research_results/live_reference/original.json",
        "research_results/live_reference/closed.json",
    }:
        return True
    return False


def _scan_sensitive_text(text: str, location: str) -> None:
    if _WINDOWS_PERSONAL_PATH.search(text) or _POSIX_PERSONAL_PATH.search(text):
        raise ReleaseValidationError(
            f"{location}: contains an absolute personal home-directory path"
        )
    for label, pattern in _CREDENTIAL_PATTERNS:
        if pattern.search(text):
            raise ReleaseValidationError(
                f"{location}: contains a credential signature ({label})"
            )


def _scan_text(text: str, location: str) -> None:
    if "\x00" in text:
        raise ReleaseValidationError(f"{location}: NUL byte in a text file")
    _scan_sensitive_text(text, location)


def _scan_office_container(data: bytes, location: str) -> None:
    expanded = 0
    try:
        with zipfile.ZipFile(io.BytesIO(data), "r") as package:
            for info in package.infolist():
                nested_name = info.filename.replace("\\", "/")
                if _denied_path_reason(nested_name):
                    raise ReleaseValidationError(
                        f"{location}: unsafe path inside document container"
                    )
                expanded += max(0, int(info.file_size))
                if expanded > _MAX_OFFICE_EXPANDED_BYTES:
                    raise ReleaseValidationError(
                        f"{location}: expanded document container is too large"
                    )
                if PurePosixPath(nested_name).suffix.casefold() not in _OFFICE_TEXT_SUFFIXES:
                    continue
                nested = package.read(info)
                try:
                    text = nested.decode("utf-8")
                except UnicodeDecodeError:
                    text = nested.decode("utf-8", errors="replace")
                _scan_text(text, f"{location}!{nested_name}")
    except (zipfile.BadZipFile, OSError) as exc:
        raise ReleaseValidationError(
            f"{location}: invalid office-document fixture ({type(exc).__name__})"
        ) from exc


def _validate_release_file(relative_path: str, data: bytes) -> None:
    reason = _denied_path_reason(relative_path)
    if reason:
        raise ReleaseValidationError(f"{relative_path}: {reason}")
    if not _allowed_release_path(relative_path):
        raise ReleaseValidationError(f"{relative_path}: path is not release-allowlisted")
    if len(data) > _MAX_FILE_BYTES:
        raise ReleaseValidationError(f"{relative_path}: file exceeds release size limit")

    suffix = PurePosixPath(relative_path).suffix.casefold()
    if suffix in _BINARY_FIXTURE_SUFFIXES:
        if not relative_path.startswith("tests/fixtures/"):
            raise ReleaseValidationError(
                f"{relative_path}: binary documents are allowed only as synthetic test fixtures"
            )
        # Scan visible byte strings even when the binary format is not fully
        # decoded. DOCX receives a second scan of its XML members below.
        _scan_sensitive_text(data.decode("latin-1"), relative_path)
        if suffix == ".docx":
            _scan_office_container(data, relative_path)
        return

    if suffix not in _TEXT_SUFFIXES:
        raise ReleaseValidationError(f"{relative_path}: unsupported release file type")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReleaseValidationError(f"{relative_path}: text is not valid UTF-8") from exc
    _scan_text(text, relative_path)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def collect_release_files(root: str | Path) -> ReleasePlan:
    """Read, validate, and freeze the exact bytes planned for an archive."""

    source_root = Path(root).resolve()
    if not source_root.is_dir():
        raise ReleaseValidationError(f"release root is not a directory: {source_root}")

    files: list[ReleaseFile] = []
    errors: list[str] = []
    for path in sorted(source_root.rglob("*"), key=lambda item: item.as_posix().casefold()):
        try:
            relative = path.relative_to(source_root).as_posix()
        except ValueError:
            continue
        if not _allowed_release_path(relative):
            continue
        if path.is_symlink():
            errors.append(f"{relative}: symlinks are not allowed in releases")
            continue
        if not path.is_file():
            continue
        try:
            resolved = path.resolve(strict=True)
            if not _is_within(resolved, source_root):
                raise ReleaseValidationError(f"{relative}: resolves outside release root")
            data = path.read_bytes()
            _validate_release_file(relative, data)
            mode = 0o755 if relative == "sisu-reader" else 0o644
            files.append(ReleaseFile(relative, data, mode))
        except (OSError, ReleaseValidationError) as exc:
            errors.append(str(exc))

    found = {item.relative_path for item in files}
    missing = sorted(REQUIRED_RELEASE_PATHS - found)
    if missing:
        errors.append("missing required release files: " + ", ".join(missing))
    total = sum(len(item.data) for item in files)
    if total > _MAX_TOTAL_BYTES:
        errors.append("allowlisted release content exceeds the total size limit")
    if errors:
        raise ReleaseValidationError("\n".join(errors))

    ordered = tuple(sorted(files, key=lambda item: item.relative_path))
    license_present = any(
        name in found for name in ("LICENSE", "LICENSE.md", "LICENSE.txt")
    )
    return ReleasePlan(ordered, total, license_present)


def _write_deterministic_zip(path: Path, files: Iterable[ReleaseFile]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.comment = b""
        for item in files:
            info = zipfile.ZipInfo(item.relative_path, date_time=_FIXED_ZIP_TIME)
            info.create_system = 3
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = (item.mode & 0xFFFF) << 16
            archive.writestr(info, item.data)


def validate_archive(path: str | Path) -> ArchiveReport:
    """Re-open an archive and enforce the same privacy and allowlist rules."""

    archive_path = Path(path).resolve()
    if not archive_path.is_file():
        raise ReleaseValidationError(f"archive does not exist: {archive_path}")

    names: list[str] = []
    total = 0
    with zipfile.ZipFile(archive_path, "r") as archive:
        if archive.comment:
            raise ReleaseValidationError("release archive must not have a comment")
        for info in archive.infolist():
            name = info.filename.replace("\\", "/")
            if info.is_dir():
                raise ReleaseValidationError(f"{name}: directory entries are not permitted")
            if info.flag_bits & 0x1:
                raise ReleaseValidationError(f"{name}: encrypted entries are not permitted")
            if info.compress_type != zipfile.ZIP_STORED:
                raise ReleaseValidationError(f"{name}: non-deterministic compression is not permitted")
            if info.date_time != _FIXED_ZIP_TIME:
                raise ReleaseValidationError(f"{name}: archive timestamp is not normalized")
            if name in names:
                raise ReleaseValidationError(f"{name}: duplicate archive entry")
            data = archive.read(info)
            _validate_release_file(name, data)
            names.append(name)
            total += len(data)
            if total > _MAX_TOTAL_BYTES:
                raise ReleaseValidationError("release archive exceeds the total size limit")

    if names != sorted(names):
        raise ReleaseValidationError("release archive entries are not deterministically ordered")
    missing = sorted(REQUIRED_RELEASE_PATHS - set(names))
    if missing:
        raise ReleaseValidationError(
            "release archive is missing required files: " + ", ".join(missing)
        )
    encoded = archive_path.read_bytes()
    license_present = any(
        name in names for name in ("LICENSE", "LICENSE.md", "LICENSE.txt")
    )
    return ArchiveReport(
        archive_path=archive_path,
        file_count=len(names),
        total_bytes=total,
        sha256=hashlib.sha256(encoded).hexdigest(),
        license_present=license_present,
    )


def build_archive(
    root: str | Path,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> ArchiveReport:
    """Build one atomic deterministic archive and validate it before install."""

    source_root = Path(root).resolve()
    target = Path(output)
    if not target.is_absolute():
        target = (source_root / target).resolve()
    else:
        target = target.resolve()
    if target.exists() and not overwrite:
        raise ReleaseValidationError(
            f"output already exists: {target}; pass overwrite=True or --force explicitly"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    plan = collect_release_files(source_root)
    temporary = target.parent / f".{target.name}.{os.getpid()}.tmp"
    try:
        _write_deterministic_zip(temporary, plan.files)
        checked = validate_archive(temporary)
        os.replace(temporary, target)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return ArchiveReport(
        archive_path=target,
        file_count=checked.file_count,
        total_bytes=checked.total_bytes,
        sha256=checked.sha256,
        license_present=checked.license_present,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sisu_reader.release",
        description="Build a deterministic source-only SISU Reader archive.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Project root to validate (defaults to the installed source tree)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("final.zip"),
        help="Archive path, relative to --root unless absolute",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the release plan without creating an archive",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly replace an existing output archive",
    )
    return parser


def _license_notice(present: bool) -> None:
    if not present:
        print(
            "warning: no LICENSE file is present; owner/legal review must define "
            "redistribution terms before external distribution",
            file=sys.stderr,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.check:
            plan = collect_release_files(args.root)
            print(
                f"Release plan is clean: {len(plan.files)} files, "
                f"{plan.total_bytes} source bytes."
            )
            _license_notice(plan.license_present)
            return 0
        report = build_archive(args.root, args.output, overwrite=args.force)
        print(
            f"Created {report.archive_path} with {report.file_count} files "
            f"({report.total_bytes} source bytes)."
        )
        print(f"SHA-256: {report.sha256}")
        _license_notice(report.license_present)
        return 0
    except (OSError, ReleaseValidationError, zipfile.BadZipFile) as exc:
        print(f"release error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ArchiveReport",
    "RELEASE_FORMAT_VERSION",
    "REQUIRED_RELEASE_PATHS",
    "ReleaseFile",
    "ReleasePlan",
    "ReleaseValidationError",
    "build_archive",
    "collect_release_files",
    "main",
    "validate_archive",
]
