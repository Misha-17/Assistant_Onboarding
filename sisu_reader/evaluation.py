"""Deterministic release evaluation for synthetic SISU Reader cases.

This module scores only observable contract properties. It does not estimate
semantic confidence and it does not claim that string checks prove entailment.
Human-reviewed benchmark labels define the expected status, phrases, resource
identities, citation cap, principal, and identifiers that must not leak.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


BENCHMARK_SCHEMA_VERSION = 1
RESULTS_SCHEMA_VERSION = 1
_ANSWER_STATUSES = frozenset({"answer", "partial", "clarify", "not_found", "error"})
_RESOURCE_ID_KEYS = (
    "resource_id",
    "source_resource_id",
    "document_revision_id",
    "document_id",
)


class EvaluationValidationError(ValueError):
    """Raised when a benchmark or result file violates the evaluation schema."""


@dataclass(frozen=True, slots=True)
class BenchmarkResource:
    resource_id: str
    resource_type: str
    title: str
    allowed_principals: tuple[str, ...]
    synthetic_text: str


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    category: str
    question: str
    principal: str
    expected_status: str
    required_claim_phrases: tuple[str, ...]
    forbidden_claim_phrases: tuple[str, ...]
    required_source_resource_ids: tuple[str, ...]
    allowed_source_resource_ids: tuple[str, ...]
    forbidden_source_resource_ids: tuple[str, ...]
    unauthorized_identifiers: tuple[str, ...]
    max_citations: int


@dataclass(frozen=True, slots=True)
class Benchmark:
    benchmark_id: str
    benchmark_version: str
    schema_version: int
    principals: tuple[str, ...]
    resources: tuple[BenchmarkResource, ...]
    cases: tuple[BenchmarkCase, ...]

    @property
    def resources_by_id(self) -> dict[str, BenchmarkResource]:
        return {item.resource_id: item for item in self.resources}

    @property
    def cases_by_id(self) -> dict[str, BenchmarkCase]:
        return {item.case_id: item for item in self.cases}


@dataclass(frozen=True, slots=True)
class EvaluationCheck:
    check: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    case_id: str
    category: str
    principal: str
    passed: bool
    checks_passed: int
    checks_total: int
    checks: tuple[EvaluationCheck, ...]


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    benchmark_id: str
    benchmark_version: str
    schema_version: int
    passed: bool
    cases_total: int
    cases_submitted: int
    cases_passed: int
    cases_failed: int
    checks_passed: int
    checks_total: int
    missing_case_ids: tuple[str, ...]
    unknown_case_ids: tuple[str, ...]
    cases: tuple[CaseEvaluation, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationValidationError(f"{location} must be a JSON object")
    return value


def _required_string(row: Mapping[str, Any], key: str, location: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise EvaluationValidationError(f"{location}.{key} must be a non-empty string")
    return value.strip()


def _string_tuple(
    row: Mapping[str, Any],
    key: str,
    location: str,
    *,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    value = row.get(key)
    if not isinstance(value, list):
        raise EvaluationValidationError(f"{location}.{key} must be a JSON array")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise EvaluationValidationError(
                f"{location}.{key}[{index}] must be a non-empty string"
            )
        clean = item.strip()
        if clean in result:
            raise EvaluationValidationError(f"{location}.{key} contains a duplicate")
        result.append(clean)
    if not allow_empty and not result:
        raise EvaluationValidationError(f"{location}.{key} cannot be empty")
    return tuple(result)


def _integer(row: Mapping[str, Any], key: str, location: str, minimum: int) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvaluationValidationError(
            f"{location}.{key} must be an integer greater than or equal to {minimum}"
        )
    return value


def parse_benchmark(payload: Mapping[str, Any]) -> Benchmark:
    """Validate a parsed benchmark and return immutable typed records."""

    root = _mapping(payload, "benchmark")
    schema_version = _integer(root, "schema_version", "benchmark", 1)
    if schema_version != BENCHMARK_SCHEMA_VERSION:
        raise EvaluationValidationError(
            f"unsupported benchmark schema {schema_version}; expected {BENCHMARK_SCHEMA_VERSION}"
        )
    if root.get("synthetic") is not True:
        raise EvaluationValidationError("benchmark.synthetic must be true")
    benchmark_id = _required_string(root, "benchmark_id", "benchmark")
    benchmark_version = _required_string(root, "benchmark_version", "benchmark")
    principals = _string_tuple(root, "principals", "benchmark", allow_empty=False)

    raw_resources = root.get("resources")
    if not isinstance(raw_resources, list) or not raw_resources:
        raise EvaluationValidationError("benchmark.resources must be a non-empty JSON array")
    resources: list[BenchmarkResource] = []
    resource_ids: set[str] = set()
    for index, raw in enumerate(raw_resources):
        location = f"benchmark.resources[{index}]"
        row = _mapping(raw, location)
        resource_id = _required_string(row, "resource_id", location)
        if resource_id in resource_ids:
            raise EvaluationValidationError(f"duplicate resource_id: {resource_id}")
        allowed = _string_tuple(row, "allowed_principals", location, allow_empty=False)
        unknown_principals = sorted(set(allowed) - set(principals))
        if unknown_principals:
            raise EvaluationValidationError(
                f"{location}.allowed_principals contains unknown principals: "
                + ", ".join(unknown_principals)
            )
        resources.append(
            BenchmarkResource(
                resource_id=resource_id,
                resource_type=_required_string(row, "resource_type", location),
                title=_required_string(row, "title", location),
                allowed_principals=allowed,
                synthetic_text=_required_string(row, "synthetic_text", location),
            )
        )
        resource_ids.add(resource_id)

    raw_cases = root.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise EvaluationValidationError("benchmark.cases must be a non-empty JSON array")
    cases: list[BenchmarkCase] = []
    case_ids: set[str] = set()
    resources_by_id = {item.resource_id: item for item in resources}
    for index, raw in enumerate(raw_cases):
        location = f"benchmark.cases[{index}]"
        row = _mapping(raw, location)
        case_id = _required_string(row, "case_id", location)
        if case_id in case_ids:
            raise EvaluationValidationError(f"duplicate case_id: {case_id}")
        principal = _required_string(row, "principal", location)
        if principal not in principals:
            raise EvaluationValidationError(f"{location}.principal is not registered")
        expected_status = _required_string(row, "expected_status", location).casefold()
        if expected_status not in _ANSWER_STATUSES:
            raise EvaluationValidationError(f"{location}.expected_status is unsupported")
        required_phrases = _string_tuple(row, "required_claim_phrases", location)
        forbidden_phrases = _string_tuple(row, "forbidden_claim_phrases", location)
        normalized_required = {_normal_text(item) for item in required_phrases}
        normalized_forbidden = {_normal_text(item) for item in forbidden_phrases}
        if normalized_required & normalized_forbidden:
            raise EvaluationValidationError(
                f"{location} has the same phrase in required and forbidden lists"
            )
        required_sources = _string_tuple(
            row, "required_source_resource_ids", location
        )
        allowed_sources = _string_tuple(row, "allowed_source_resource_ids", location)
        forbidden_sources = _string_tuple(
            row, "forbidden_source_resource_ids", location
        )
        all_case_source_ids = set(required_sources) | set(allowed_sources) | set(
            forbidden_sources
        )
        unknown_sources = sorted(all_case_source_ids - resource_ids)
        if unknown_sources:
            raise EvaluationValidationError(
                f"{location} refers to unknown resource IDs: " + ", ".join(unknown_sources)
            )
        if not set(required_sources).issubset(allowed_sources):
            raise EvaluationValidationError(
                f"{location}.required_source_resource_ids must be allowed"
            )
        if set(allowed_sources) & set(forbidden_sources):
            raise EvaluationValidationError(
                f"{location} has resource IDs in both allowed and forbidden lists"
            )
        unauthorized_allowed = sorted(
            resource_id
            for resource_id in allowed_sources
            if principal not in resources_by_id[resource_id].allowed_principals
        )
        if unauthorized_allowed:
            raise EvaluationValidationError(
                f"{location} allows resources unavailable to its principal: "
                + ", ".join(unauthorized_allowed)
            )
        max_citations = _integer(row, "max_citations", location, 0)
        if max_citations < len(required_sources):
            raise EvaluationValidationError(
                f"{location}.max_citations is smaller than its required source count"
            )
        cases.append(
            BenchmarkCase(
                case_id=case_id,
                category=_required_string(row, "category", location),
                question=_required_string(row, "question", location),
                principal=principal,
                expected_status=expected_status,
                required_claim_phrases=required_phrases,
                forbidden_claim_phrases=forbidden_phrases,
                required_source_resource_ids=required_sources,
                allowed_source_resource_ids=allowed_sources,
                forbidden_source_resource_ids=forbidden_sources,
                unauthorized_identifiers=_string_tuple(
                    row, "unauthorized_identifiers", location
                ),
                max_citations=max_citations,
            )
        )
        case_ids.add(case_id)

    return Benchmark(
        benchmark_id=benchmark_id,
        benchmark_version=benchmark_version,
        schema_version=schema_version,
        principals=principals,
        resources=tuple(resources),
        cases=tuple(cases),
    )


def load_benchmark(path: str | Path) -> Benchmark:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationValidationError(
            f"could not load benchmark {source.name}: {type(exc).__name__}"
        ) from exc
    return parse_benchmark(_mapping(payload, "benchmark"))


def _normal_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(normalized.split()).casefold()


def _citation_rows(answer: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw = answer.get("sources", answer.get("citations", []))
    if not isinstance(raw, list):
        return ()
    return tuple(item for item in raw if isinstance(item, Mapping))


def _citation_identity(row: Mapping[str, Any]) -> str:
    for key in _RESOURCE_ID_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _check(name: str, passed: bool, detail: str) -> EvaluationCheck:
    return EvaluationCheck(check=name, passed=bool(passed), detail=detail)


def _score_case(
    case: BenchmarkCase,
    submitted: Mapping[str, Any] | None,
) -> CaseEvaluation:
    checks: list[EvaluationCheck] = []
    if submitted is None:
        checks.append(_check("result_present", False, "No result was supplied."))
        return CaseEvaluation(
            case_id=case.case_id,
            category=case.category,
            principal=case.principal,
            passed=False,
            checks_passed=0,
            checks_total=1,
            checks=tuple(checks),
        )

    principal = submitted.get("principal")
    checks.append(
        _check(
            "principal",
            principal == case.principal,
            "Principal matches the benchmark case."
            if principal == case.principal
            else "Result principal does not match the benchmark case.",
        )
    )
    raw_answer = submitted.get("answer")
    if not isinstance(raw_answer, Mapping):
        checks.append(_check("answer_object", False, "answer must be a JSON object."))
        passed = all(item.passed for item in checks)
        return CaseEvaluation(
            case_id=case.case_id,
            category=case.category,
            principal=case.principal,
            passed=passed,
            checks_passed=sum(item.passed for item in checks),
            checks_total=len(checks),
            checks=tuple(checks),
        )
    answer = raw_answer
    status = str(answer.get("status") or "").strip().casefold()
    checks.append(
        _check(
            "status",
            status == case.expected_status,
            f"Expected {case.expected_status!r}; received {status or '<missing>'!r}.",
        )
    )
    text = _normal_text(answer.get("text"))
    for index, phrase in enumerate(case.required_claim_phrases, 1):
        present = _normal_text(phrase) in text
        checks.append(
            _check(
                f"required_claim_phrase_{index}",
                present,
                "Required reviewed phrase is present."
                if present
                else f"Missing required reviewed phrase: {phrase!r}.",
            )
        )
    for index, phrase in enumerate(case.forbidden_claim_phrases, 1):
        absent = _normal_text(phrase) not in text
        checks.append(
            _check(
                f"forbidden_claim_phrase_{index}",
                absent,
                "Forbidden reviewed phrase is absent."
                if absent
                else f"Forbidden reviewed phrase appeared: {phrase!r}.",
            )
        )

    raw_sources = answer.get("sources", answer.get("citations", []))
    source_shape_ok = isinstance(raw_sources, list) and all(
        isinstance(item, Mapping) for item in raw_sources
    )
    rows = _citation_rows(answer)
    identities = tuple(_citation_identity(item) for item in rows)
    checks.append(
        _check(
            "citation_shape",
            source_shape_ok,
            "Citations are JSON objects in an array."
            if source_shape_ok
            else "sources/citations must be an array of JSON objects.",
        )
    )
    identity_ok = all(identities)
    checks.append(
        _check(
            "citation_identities_present",
            identity_ok,
            "Every citation carries a stable resource identity."
            if identity_ok
            else "One or more citations lack a stable resource identity.",
        )
    )
    for index, resource_id in enumerate(case.required_source_resource_ids, 1):
        present = resource_id in identities
        checks.append(
            _check(
                f"required_source_{index}",
                present,
                f"Required source {resource_id!r} is present."
                if present
                else f"Required source {resource_id!r} is missing.",
            )
        )
    unexpected = sorted(
        {item for item in identities if item and item not in case.allowed_source_resource_ids}
    )
    checks.append(
        _check(
            "only_allowed_sources",
            not unexpected,
            "All cited resources are allowed for this case."
            if not unexpected
            else "Unexpected cited resources: " + ", ".join(unexpected),
        )
    )
    for index, resource_id in enumerate(case.forbidden_source_resource_ids, 1):
        absent = resource_id not in identities
        checks.append(
            _check(
                f"forbidden_source_{index}",
                absent,
                f"Forbidden source {resource_id!r} is absent."
                if absent
                else f"Forbidden source {resource_id!r} was cited.",
            )
        )
    citation_cap_ok = len(rows) <= case.max_citations
    checks.append(
        _check(
            "citation_cap",
            citation_cap_ok,
            f"Citation count {len(rows)} is within cap {case.max_citations}."
            if citation_cap_ok
            else f"Citation count {len(rows)} exceeds cap {case.max_citations}.",
        )
    )

    serialized_answer = _normal_text(
        json.dumps(answer, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    for index, identifier in enumerate(case.unauthorized_identifiers, 1):
        absent = _normal_text(identifier) not in serialized_answer
        checks.append(
            _check(
                f"unauthorized_identifier_{index}",
                absent,
                "Restricted identifier is absent from the complete answer payload."
                if absent
                else "A restricted identifier appeared in the answer payload.",
            )
        )

    passed = all(item.passed for item in checks)
    return CaseEvaluation(
        case_id=case.case_id,
        category=case.category,
        principal=case.principal,
        passed=passed,
        checks_passed=sum(item.passed for item in checks),
        checks_total=len(checks),
        checks=tuple(checks),
    )


def score_results(benchmark: Benchmark, payload: Mapping[str, Any]) -> EvaluationReport:
    """Score one versioned result envelope against a validated benchmark."""

    root = _mapping(payload, "results")
    schema_version = _integer(root, "schema_version", "results", 1)
    if schema_version != RESULTS_SCHEMA_VERSION:
        raise EvaluationValidationError(
            f"unsupported result schema {schema_version}; expected {RESULTS_SCHEMA_VERSION}"
        )
    if _required_string(root, "benchmark_id", "results") != benchmark.benchmark_id:
        raise EvaluationValidationError("results.benchmark_id does not match the benchmark")
    if _required_string(root, "benchmark_version", "results") != benchmark.benchmark_version:
        raise EvaluationValidationError(
            "results.benchmark_version does not match the benchmark"
        )
    raw_results = root.get("results")
    if not isinstance(raw_results, list):
        raise EvaluationValidationError("results.results must be a JSON array")

    submitted: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(raw_results):
        location = f"results.results[{index}]"
        row = _mapping(raw, location)
        case_id = _required_string(row, "case_id", location)
        if case_id in submitted:
            raise EvaluationValidationError(f"duplicate submitted case_id: {case_id}")
        submitted[case_id] = row

    expected_ids = set(benchmark.cases_by_id)
    submitted_ids = set(submitted)
    missing = tuple(sorted(expected_ids - submitted_ids))
    unknown = tuple(sorted(submitted_ids - expected_ids))
    cases = tuple(
        _score_case(case, submitted.get(case.case_id)) for case in benchmark.cases
    )
    checks_passed = sum(item.checks_passed for item in cases)
    checks_total = sum(item.checks_total for item in cases)
    cases_passed = sum(item.passed for item in cases)
    passed = not missing and not unknown and cases_passed == len(cases)
    return EvaluationReport(
        benchmark_id=benchmark.benchmark_id,
        benchmark_version=benchmark.benchmark_version,
        schema_version=RESULTS_SCHEMA_VERSION,
        passed=passed,
        cases_total=len(benchmark.cases),
        cases_submitted=len(submitted_ids & expected_ids),
        cases_passed=cases_passed,
        cases_failed=len(cases) - cases_passed,
        checks_passed=checks_passed,
        checks_total=checks_total,
        missing_case_ids=missing,
        unknown_case_ids=unknown,
        cases=cases,
    )


def load_results(path: str | Path) -> Mapping[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationValidationError(
            f"could not load results {source.name}: {type(exc).__name__}"
        ) from exc
    return _mapping(payload, "results")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sisu_reader.evaluation",
        description="Score Answer-like JSON against a synthetic deterministic benchmark.",
    )
    parser.add_argument("benchmark", type=Path, help="Versioned benchmark JSON")
    parser.add_argument("results", type=Path, help="Answer result envelope JSON")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument(
        "--compact", action="store_true", help="Print compact rather than indented JSON"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        benchmark = load_benchmark(args.benchmark)
        report = score_results(benchmark, load_results(args.results))
        payload = report.to_dict()
        if args.output is not None:
            _write_json(args.output, payload)
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=None if args.compact else 2,
                separators=(",", ":") if args.compact else None,
            )
        )
        return 0 if report.passed else 1
    except (OSError, EvaluationValidationError) as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "RESULTS_SCHEMA_VERSION",
    "Benchmark",
    "BenchmarkCase",
    "BenchmarkResource",
    "CaseEvaluation",
    "EvaluationCheck",
    "EvaluationReport",
    "EvaluationValidationError",
    "load_benchmark",
    "load_results",
    "main",
    "parse_benchmark",
    "score_results",
]

