from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import Config
from .session import Session


_PROGRESS_LABELS = {
    "authorizing": "checking access",
    "exact_search": "checking exact names and phrases",
    "screening": "screening document manifests",
    "reading": "reading selected documents",
    "adaptive_wave": "starting another evidence search",
    "evidence_review": "checking remaining evidence gaps",
    "synthesizing": "writing from the evidence",
    "claim_review": "checking answer claims against their sources",
    "binding_citations": "binding exact citations",
}


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _jsonable(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        return "<truncated>"
    if is_dataclass(value):
        return _jsonable(asdict(value), depth=depth + 1)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item, depth=depth + 1)
            for key, item in list(value.items())[:200]
        }
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item, depth=depth + 1) for item in list(value)[:200]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _engine(config: Config) -> Any:
    # Kept lazy so `--help` and syntax diagnostics remain useful even if an
    # optional local runtime dependency is unavailable.
    from .engine import SisuReader

    return SisuReader(config)


def _close(engine: Any) -> None:
    try:
        engine.close()
    except Exception:
        pass


def _friendly_error(exc: BaseException, config: Config) -> tuple[str, str]:
    raw = " ".join(str(exc).split())
    folded = raw.casefold()
    name = type(exc).__name__
    if isinstance(exc, FileNotFoundError) or "index" in folded and "not found" in folded:
        return (
            "The local document index is missing.",
            'Run `sisu-reader index "C:\\path\\to\\documents"`, then try again.',
        )
    if "model" in folded and any(word in folded for word in ("missing", "not found", "unknown", "unavailable", "pull")):
        return (
            f"The local model {config.model!r} is not installed.",
            f"Run `ollama pull {config.model}`, then try again.",
        )
    if any(word in folded for word in ("connection refused", "failed to establish", "ollama")):
        return (
            "The Ollama service is not reachable on this computer.",
            "Start Ollama, or run `ollama serve`, then try again.",
        )
    detail = raw[:500] if raw else name
    return (f"The local engine could not start: {detail}", "Run `sisu-reader doctor` for details.")


class _ProgressPrinter:
    def __init__(self) -> None:
        self.last_stage = ""
        self.started = time.monotonic()

    def __call__(self, event: Any = None, *args: Any, **kwargs: Any) -> None:
        del args
        stage = ""
        if isinstance(event, str):
            stage = event
        elif isinstance(event, Mapping):
            stage = str(event.get("stage") or event.get("status") or event.get("name") or "")
        elif event is not None:
            stage = str(
                getattr(event, "stage", "")
                or getattr(event, "status", "")
                or getattr(event, "name", "")
            )
        if not stage:
            stage = str(kwargs.get("stage") or kwargs.get("status") or kwargs.get("name") or "")
        stage = stage.strip().casefold().replace("-", "_").replace(" ", "_")
        if stage not in _PROGRESS_LABELS or stage == self.last_stage:
            return
        self.last_stage = stage
        elapsed = time.monotonic() - self.started
        print(f"  {elapsed:5.1f}s | {_PROGRESS_LABELS[stage]}", file=sys.stderr, flush=True)


def _print_sources(answer: Any) -> None:
    sources = _value(answer, "sources", ())
    if isinstance(sources, (str, bytes, Mapping)):
        return
    try:
        rows = tuple(sources or ())
    except TypeError:
        return
    if not rows:
        return
    print("\nSources:")
    for index, source in enumerate(rows, 1):
        source_id = str(_value(source, "source_id", "") or f"S{index}")
        title = str(_value(source, "title", "Untitled source"))
        locator = str(_value(source, "locator", "")).strip()
        uri = str(_value(source, "uri", "")).strip()
        suffix = f" | {locator}" if locator else ""
        if uri:
            suffix += f" | {uri}"
        print(f"  [{source_id}] {title}{suffix}")


def _print_answer(answer: Any, *, debug: bool = False) -> None:
    text = str(_value(answer, "text", "") or "").strip()
    status = str(_value(answer, "status", "answer") or "answer")
    print(text or f"[{status}] No answer text was returned.")
    _print_sources(answer)

    warnings = _value(answer, "warnings", ())
    if not isinstance(warnings, (str, bytes, Mapping)):
        try:
            rows = tuple(warnings or ())
        except TypeError:
            rows = ()
        if rows:
            print("\nNotes:")
            for warning in rows:
                print(f"  - {' '.join(str(warning).split())}")

    trace_path = str(_value(answer, "trace_path", "") or "").strip()
    if trace_path:
        print(f"\nTrace: {trace_path}")

    coverage = _value(answer, "coverage", {})
    if debug:
        payload = {
            "status": status,
            "timings": _value(answer, "timings", {}),
            "coverage": coverage,
            "debug": _value(answer, "debug", {}),
        }
        print("\nDebug:")
        print(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2))


def _load(config: Config) -> Any:
    engine = _engine(config)
    try:
        engine.load()
    except Exception:
        _close(engine)
        raise
    return engine


def _ask(engine: Any, question: str, session: Session, *, progress: bool = True) -> Any:
    callback = _ProgressPrinter() if progress else None
    answer = engine.ask(question, session=session, mode="auto", progress=callback)
    session.record(question, answer)
    return answer


def _command_index(config: Config, paths: Sequence[str]) -> int:
    resolved: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            print(f"error: document path does not exist: {path}", file=sys.stderr)
            return 2
        resolved.append(path)

    engine = _engine(config)
    try:
        report = engine.rebuild(resolved)
    except Exception as exc:
        message, action = _friendly_error(exc, config)
        print(f"error: {message}\n{action}", file=sys.stderr)
        return 1
    finally:
        _close(engine)

    documents = _value(report, "documents", 0)
    sections = _value(report, "sections", 0)
    blocks = _value(report, "blocks", _value(report, "chunks", 0))
    elapsed = float(_value(report, "elapsed_s", 0.0) or 0.0)
    revision = str(
        _value(report, "snapshot_id", "")
        or _value(report, "corpus_revision", "")
        or ""
    )
    print(
        f"Indexed {documents} documents, {sections} sections, and "
        f"{blocks} source blocks in {elapsed:.1f}s."
    )
    if revision:
        print(f"Corpus revision: {revision}")
    for warning in tuple(_value(report, "warnings", ()) or ()):
        print(f"warning: {' '.join(str(warning).split())}", file=sys.stderr)
    return 0


def _command_doctor(config: Config) -> int:
    print(f"Workspace: {config.workspace_dir}")
    print(f"Model: {config.model}")
    print(f"Ollama: {config.ollama_url}")
    engine = None
    try:
        engine = _load(config)
        status = engine.status()
        print(json.dumps(_jsonable(status), ensure_ascii=False, indent=2))
        ready = _value(status, "ready", True)
        if ready is False:
            print("error: the engine reported that it is not ready", file=sys.stderr)
            return 1
        print("Doctor: ready.")
        return 0
    except Exception as exc:
        message, action = _friendly_error(exc, config)
        print(f"error: {message}\n{action}", file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            _close(engine)


def _command_ask(config: Config, question: str, *, debug: bool) -> int:
    engine = None
    try:
        engine = _load(config)
        answer = _ask(
            engine,
            question,
            Session(max_turns=config.session_turns),
        )
        _print_answer(answer, debug=debug)
        return 0 if str(_value(answer, "status", "error")) != "error" else 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        message, action = _friendly_error(exc, config)
        print(f"error: {message}\n{action}", file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            _close(engine)


def _command_chat(config: Config, *, debug: bool) -> int:
    engine = None
    try:
        engine = _load(config)
    except Exception as exc:
        message, action = _friendly_error(exc, config)
        print(f"error: {message}\n{action}", file=sys.stderr)
        return 1

    session = Session(max_turns=config.session_turns)
    print(
        f"SISU Reader | {config.model}. Commands: /new, /sources, /status, /quit"
    )
    last_answer: Any = None
    try:
        while True:
            try:
                question = input("you> ").strip()
            except EOFError:
                print()
                break
            if not question:
                continue
            command = question.casefold()
            if command in {"/quit", "/exit", "/q"}:
                break
            if command == "/new":
                session.reset()
                last_answer = None
                print("Started a new conversation.")
                continue
            if command == "/sources":
                if last_answer is None:
                    print("No cited sources yet.")
                else:
                    _print_sources(last_answer)
                continue
            if command == "/status":
                print(json.dumps(_jsonable(engine.status()), ensure_ascii=False, indent=2))
                continue
            try:
                last_answer = _ask(engine, question, session)
                _print_answer(last_answer, debug=debug)
            except KeyboardInterrupt:
                print("\nAnswer interrupted. The conversation is still open.", file=sys.stderr)
            except Exception as exc:
                message, action = _friendly_error(exc, config)
                print(f"error: {message}\n{action}", file=sys.stderr)
        return 0
    finally:
        _close(engine)


def _data_services(config: Config) -> tuple[Any, Any, Any]:
    from .access import AccessManager
    from .resources import ResourceRegistry
    from .store import CorpusStore

    store = CorpusStore(config).load()
    access = AccessManager(config)
    return store, access, ResourceRegistry(config, access)


def _print_json(value: Any) -> None:
    print(json.dumps(_jsonable(value), ensure_ascii=False, indent=2))


def _command_person(config: Config, args: argparse.Namespace) -> int:
    store = None
    try:
        store, _access, registry = _data_services(config)
        principal = config.principal_id
        if args.person_action == "add":
            result = registry.create_person(
                principal,
                args.name,
                person_id=args.id or "",
                aliases=tuple(args.alias or ()),
                organization=args.organization or "",
                email=args.email or "",
                classification=args.classification,
            )
        elif args.person_action == "update":
            result = registry.update_person(
                principal,
                args.person_id,
                name=args.name,
                aliases=tuple(args.alias) if args.alias is not None else None,
                organization=args.organization,
                email=args.email,
                classification=args.classification,
            )
        elif args.person_action == "deactivate":
            registry.deactivate_person(principal, args.person_id)
            result = {"deactivated": args.person_id}
        else:
            result = registry.list_people(
                principal,
                args.query or "",
                include_inactive=bool(args.history),
            )
        _print_json(result)
        return 0
    except Exception as exc:
        print(f"error: {' '.join(str(exc).split()) or type(exc).__name__}", file=sys.stderr)
        return 1
    finally:
        if store is not None:
            store.close()


def _command_role(config: Config, args: argparse.Namespace) -> int:
    store = None
    try:
        store, _access, registry = _data_services(config)
        principal = config.principal_id
        if args.role_action == "add":
            result = registry.add_role(
                principal,
                args.person_id,
                args.role_name,
                role_id=args.id or "",
                organization=args.organization or "",
                scope=args.scope or "",
                responsibility=args.responsibility or "",
                target_person_id=args.target_person or "",
                valid_from=args.valid_from or "",
                valid_until=args.valid_until or "",
                status=args.status,
                provenance_kind=args.provenance,
                source_document_revision_id=args.source_document or "",
                source_block_id=args.source_block or "",
                provenance_note=args.note or "",
                classification=args.classification,
            )
        elif args.role_action == "update":
            changes = {
                key: value
                for key, value in {
                    "person_id": args.person_id,
                    "role_name": args.role_name,
                    "new_role_id": args.new_id,
                    "organization": args.organization,
                    "scope": args.scope,
                    "responsibility": args.responsibility,
                    "target_person_id": args.target_person,
                    "valid_from": args.valid_from,
                    "valid_until": args.valid_until,
                    "status": args.status,
                    "provenance_note": args.note,
                    "classification": args.classification,
                }.items()
                if value is not None
            }
            result = registry.update_role(principal, args.role_id, **changes)
        elif args.role_action == "end":
            result = registry.end_role(
                principal, args.role_id, ended_at=args.at or ""
            )
        elif args.role_action == "deactivate":
            registry.deactivate_role(principal, args.role_id)
            result = {"deactivated": args.role_id}
        else:
            result = registry.search_roles(
                principal,
                args.query or "",
                include_history=bool(args.history),
                as_of=args.as_of or "",
                limit=args.limit,
            )
        _print_json(result)
        return 0
    except Exception as exc:
        print(f"error: {' '.join(str(exc).split()) or type(exc).__name__}", file=sys.stderr)
        return 1
    finally:
        if store is not None:
            store.close()


def _command_video(config: Config, args: argparse.Namespace) -> int:
    store = None
    try:
        store, _access, registry = _data_services(config)
        principal = config.principal_id
        if args.video_action == "import":
            source = Path(args.path).expanduser().resolve(strict=True)
            payload = json.loads(source.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                result = tuple(registry.import_video(principal, item) for item in payload)
            else:
                result = registry.import_video(principal, payload)
        elif args.video_action == "deactivate":
            registry.deactivate_video(principal, args.video_id)
            result = {"deactivated": args.video_id}
        else:
            rows = registry.search_videos(principal, args.query or "", limit=args.limit)
            result = tuple({
                "video": video,
                "matching_chapter": chapter,
                "match_score": score,
            } for video, chapter, score in rows)
        _print_json(result)
        return 0
    except Exception as exc:
        print(f"error: {' '.join(str(exc).split()) or type(exc).__name__}", file=sys.stderr)
        return 1
    finally:
        if store is not None:
            store.close()


def _command_access(config: Config, args: argparse.Namespace) -> int:
    store = None
    try:
        store, access, _registry = _data_services(config)
        actor = config.principal_id
        action = args.access_action
        required_permission = "audit.read" if action == "audit" else "access.manage"
        if not access.has_permission(actor, required_permission):
            raise PermissionError("access administration is unavailable")
        if action == "user-add":
            access.create_user(args.user_id, args.name, actor_user_id=actor)
            access.assign_system_role(
                args.user_id, args.system_role, actor_user_id=actor
            )
            access.grant(
                args.user_id,
                "*",
                "*",
                "question.ask",
                actor_user_id=actor,
            )
            result = access.principal_snapshot(args.user_id)
        elif action == "user-deactivate":
            access.deactivate_user(args.user_id, actor_user_id=actor)
            result = {"deactivated": args.user_id}
        elif action == "group-add":
            access.create_group(args.group_id, args.name, actor_user_id=actor)
            result = {"group_id": args.group_id}
        elif action == "member-add":
            access.add_group_member(
                args.group_id,
                args.user_id,
                valid_from=args.valid_from,
                valid_until=args.valid_until,
                actor_user_id=actor,
            )
            result = {"group_id": args.group_id, "user_id": args.user_id}
        elif action == "member-remove":
            access.remove_group_member(
                args.group_id, args.user_id, actor_user_id=actor
            )
            result = {"removed": f"{args.group_id}/{args.user_id}"}
        elif action == "role-assign":
            access.assign_system_role(
                args.principal_id,
                args.system_role,
                principal_type=args.principal_type,
                valid_from=args.valid_from,
                valid_until=args.valid_until,
                actor_user_id=actor,
            )
            result = {"assigned": args.system_role, "principal": args.principal_id}
        elif action == "grant":
            result = {"grant_id": access.grant(
                args.principal_id,
                args.resource_type,
                args.resource_id,
                args.permission,
                principal_type=args.principal_type,
                effect=args.effect,
                valid_from=args.valid_from,
                valid_until=args.valid_until,
                actor_user_id=actor,
            )}
        elif action == "revoke":
            access.revoke_grant(args.grant_id, actor_user_id=actor)
            result = {"revoked": args.grant_id}
        elif action == "audit":
            result = access.audit_listing(limit=args.limit)
        elif action == "resource-list":
            result = access.resource_listing(
                actor,
                resource_type=args.resource_type or "",
                limit=args.limit,
            )
        else:
            result = access.principal_snapshot(args.user_id)
        _print_json(result)
        return 0
    except Exception as exc:
        print(f"error: {' '.join(str(exc).split()) or type(exc).__name__}", file=sys.stderr)
        return 1
    finally:
        if store is not None:
            store.close()


def _command_ui(config: Config, *, no_browser: bool) -> int:
    from .web import run_web

    try:
        run_web(config, open_browser=not no_browser)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        message, action = _friendly_error(exc, config)
        print(f"error: {message}\n{action}", file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sisu-reader",
        description="Private, model-led research over local documents.",
    )
    parser.add_argument("--model", help="Ollama model name (overrides SISU_READER_MODEL)")
    parser.add_argument("--workspace", type=Path, help="Use an isolated index, traces and learning database")
    parser.add_argument(
        "--user",
        help="Trusted local user ID (overrides SISU_READER_USER_ID for this process)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    from .learning_cli import add_learning_parser
    add_learning_parser(subparsers)

    index = subparsers.add_parser("index", help="Build or replace the local document index")
    index.add_argument("paths", nargs="+", metavar="PATH")

    subparsers.add_parser("doctor", help="Check the index, local model, and engine")

    ask = subparsers.add_parser("ask", help="Ask one question")
    ask.add_argument("question", nargs="+", metavar="QUESTION")
    ask.add_argument("--debug", action="store_true", help="Print coverage and diagnostics")

    chat = subparsers.add_parser("chat", help="Start an interactive terminal conversation")
    chat.add_argument("--debug", action="store_true", help="Print coverage and diagnostics")

    ui = subparsers.add_parser("ui", help="Start the private loopback web interface")
    ui.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically")

    person = subparsers.add_parser("person", help="Manage the protected people directory")
    person_actions = person.add_subparsers(dest="person_action", required=True)
    person_add = person_actions.add_parser("add", help="Register a person")
    person_add.add_argument("name")
    person_add.add_argument("--id")
    person_add.add_argument("--alias", action="append")
    person_add.add_argument("--organization")
    person_add.add_argument("--email")
    person_add.add_argument(
        "--classification",
        default="internal",
        choices=("public", "internal", "confidential", "restricted"),
    )
    person_list = person_actions.add_parser("list", help="List visible people")
    person_list.add_argument("query", nargs="?")
    person_list.add_argument("--history", action="store_true")
    person_update = person_actions.add_parser("update", help="Update a person")
    person_update.add_argument("person_id")
    person_update.add_argument("--name")
    person_update.add_argument("--alias", action="append", default=None)
    person_update.add_argument("--organization")
    person_update.add_argument("--email")
    person_update.add_argument(
        "--classification",
        choices=("public", "internal", "confidential", "restricted"),
    )
    person_deactivate = person_actions.add_parser("deactivate", help="Deactivate a person")
    person_deactivate.add_argument("person_id")

    role = subparsers.add_parser("role", help="Manage persistent organizational roles")
    role_actions = role.add_subparsers(dest="role_action", required=True)
    role_add = role_actions.add_parser("add", help="Register an organizational role")
    role_add.add_argument("person_id")
    role_add.add_argument("role_name")
    role_add.add_argument("--id")
    role_add.add_argument("--organization")
    role_add.add_argument("--scope")
    role_add.add_argument("--responsibility")
    role_add.add_argument("--target-person")
    role_add.add_argument("--valid-from")
    role_add.add_argument("--valid-until")
    role_add.add_argument("--status", default="asserted", choices=("asserted", "disputed"))
    role_add.add_argument("--provenance", default="manual", choices=("manual", "document", "imported"))
    role_add.add_argument("--source-document")
    role_add.add_argument("--source-block")
    role_add.add_argument("--note")
    role_add.add_argument(
        "--classification",
        default="internal",
        choices=("public", "internal", "confidential", "restricted"),
    )
    role_list = role_actions.add_parser("list", help="Search visible organizational roles")
    role_list.add_argument("query", nargs="?")
    role_list.add_argument("--history", action="store_true")
    role_list.add_argument("--as-of")
    role_list.add_argument("--limit", type=int, default=50)
    role_update = role_actions.add_parser("update", help="Create a successor role record")
    role_update.add_argument("role_id")
    role_update.add_argument("--person-id")
    role_update.add_argument("--role-name")
    role_update.add_argument("--new-id")
    role_update.add_argument("--organization")
    role_update.add_argument("--scope")
    role_update.add_argument("--responsibility")
    role_update.add_argument("--target-person")
    role_update.add_argument("--valid-from")
    role_update.add_argument("--valid-until")
    role_update.add_argument("--status", choices=("asserted", "disputed"))
    role_update.add_argument("--note")
    role_update.add_argument(
        "--classification",
        choices=("public", "internal", "confidential", "restricted"),
    )
    role_end = role_actions.add_parser("end", help="End a role without deleting history")
    role_end.add_argument("role_id")
    role_end.add_argument("--at")
    role_deactivate = role_actions.add_parser("deactivate", help="Retract a role")
    role_deactivate.add_argument("role_id")
    video = subparsers.add_parser("video", help="Manage metadata-only video records")
    video_actions = video.add_subparsers(dest="video_action", required=True)
    video_import = video_actions.add_parser("import", help="Import supplied JSON metadata")
    video_import.add_argument("path")
    video_list = video_actions.add_parser("list", help="Search visible video metadata")
    video_list.add_argument("query", nargs="?")
    video_list.add_argument("--limit", type=int, default=20)
    video_deactivate = video_actions.add_parser("deactivate", help="Deactivate a video record")
    video_deactivate.add_argument("video_id")

    access = subparsers.add_parser("access", help="Administer local users and grants")
    access_actions = access.add_subparsers(dest="access_action", required=True)
    user_add = access_actions.add_parser("user-add")
    user_add.add_argument("user_id")
    user_add.add_argument("name")
    user_add.add_argument(
        "--system-role",
        default="viewer",
        choices=("administrator", "corpus_manager", "researcher", "viewer", "guest", "trace_auditor"),
    )
    user_deactivate = access_actions.add_parser("user-deactivate")
    user_deactivate.add_argument("user_id")
    user_show = access_actions.add_parser("user-show")
    user_show.add_argument("user_id")
    group_add = access_actions.add_parser("group-add")
    group_add.add_argument("group_id")
    group_add.add_argument("name")
    member_add = access_actions.add_parser("member-add")
    member_add.add_argument("group_id")
    member_add.add_argument("user_id")
    member_add.add_argument("--valid-from")
    member_add.add_argument("--valid-until")
    member_remove = access_actions.add_parser("member-remove")
    member_remove.add_argument("group_id")
    member_remove.add_argument("user_id")
    role_assign = access_actions.add_parser("role-assign")
    role_assign.add_argument("principal_id")
    role_assign.add_argument("system_role")
    role_assign.add_argument("--principal-type", default="user", choices=("user", "group"))
    role_assign.add_argument("--valid-from")
    role_assign.add_argument("--valid-until")
    grant = access_actions.add_parser("grant")
    grant.add_argument("principal_id")
    grant.add_argument("resource_type")
    grant.add_argument("resource_id")
    grant.add_argument("permission")
    grant.add_argument("--principal-type", default="user", choices=("user", "group"))
    grant.add_argument("--effect", default="allow", choices=("allow", "deny"))
    grant.add_argument("--valid-from")
    grant.add_argument("--valid-until")
    revoke = access_actions.add_parser("revoke")
    revoke.add_argument("grant_id")
    audit = access_actions.add_parser("audit")
    audit.add_argument("--limit", type=int, default=100)
    resource_list = access_actions.add_parser("resource-list")
    resource_list.add_argument("--type", dest="resource_type")
    resource_list.add_argument("--limit", type=int, default=200)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = Config.load()
        if args.workspace:
            config = config.with_overrides(workspace_dir=args.workspace.resolve())
        if args.model:
            config = config.with_overrides(model=args.model.strip())
        if args.user:
            user_id = args.user.strip()
            if not user_id:
                raise ValueError("--user cannot be empty")
            config = config.with_overrides(principal_id=user_id)
    except (OSError, ValueError) as exc:
        print(f"error: invalid configuration: {exc}", file=sys.stderr)
        return 2

    if args.command == "index":
        return _command_index(config, args.paths)
    if args.command == "learning":
        from .learning_cli import command_learning
        return command_learning(config, args)
    if args.command == "doctor":
        return _command_doctor(config)
    if args.command == "ask":
        return _command_ask(config, " ".join(args.question).strip(), debug=args.debug)
    if args.command == "chat":
        return _command_chat(config, debug=args.debug)
    if args.command == "ui":
        return _command_ui(config, no_browser=args.no_browser)
    if args.command == "person":
        return _command_person(config, args)
    if args.command == "role":
        return _command_role(config, args)
    if args.command == "video":
        return _command_video(config, args)
    if args.command == "access":
        return _command_access(config, args)
    return 2


__all__ = ["main"]
