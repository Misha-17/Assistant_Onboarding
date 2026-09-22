"""Reproducible synthetic evaluation of reference-context preservation.

This measures blocks delivered to a hypothetical downstream consumer and whether
incomplete context is flagged. It does not call an LLM or measure answer quality.
The oracle is recorded while fixtures are constructed, independently of the
reference resolver under test.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Sequence

from .models import Section, SourceBlock
from .reference_closure import ReferenceIndex, pack_closed


SCHEMA_VERSION = "sisu-reference-robustness-v1"
DEFAULT_SEED = 20260914
METHODS = (
    "original_selected_blocks",
    "two_hop_expansion",
    "transitive_closure_with_delivery_audit",
    "closure_with_synthesis_stripping",
    "always_abstain",
)


@dataclass(frozen=True)
class ReferenceCase:
    case_id: str
    category: str
    mutation: str
    budget_regime: str
    chain_hops: int
    blocks: tuple[SourceBlock, ...]
    sections: tuple[Section, ...]
    seed_block_ids: tuple[str, ...]
    required_block_ids: tuple[str, ...]
    oracle_unresolved: bool
    oracle_reason: str
    budget: int


def packet_cost(blocks: Sequence[SourceBlock]) -> int:
    """Deterministic surrogate tokens, including an eight-token block wrapper.

    This is a budget unit, not a claim about any model's tokenizer or billing.
    """
    return sum(block.token_estimate + 8 for block in blocks)


def _block(block_id: str, document_id: str, section_id: str, text: str) -> SourceBlock:
    return SourceBlock(
        block_id=block_id,
        document_revision_id=document_id,
        section_id=section_id,
        ordinal=0,
        kind="paragraph",
        locator=f"synthetic:{block_id}",
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        canonical_char_start=0,
        canonical_char_end=len(text),
        token_estimate=(len(text) + 3) // 4,
    )


def _section(section_id: str, document_id: str, number: int, title: str) -> Section:
    return Section(
        section_id=section_id,
        document_revision_id=document_id,
        parent_section_id=None,
        ordinal=0,
        depth=1,
        heading=f"Section {number}: {title}",
        section_path=f"Section {number}: {title}",
        locator=f"synthetic:{section_id}",
        first_block_ordinal=0,
        last_block_ordinal=0,
        token_estimate=0,
    )


def _assemble(
    *,
    case_id: str,
    category: str,
    mutation: str,
    budget_regime: str,
    chain_hops: int,
    entries: list[tuple[Section, SourceBlock]],
    seed_ids: tuple[str, ...],
    required_ids: tuple[str, ...],
    oracle_unresolved: bool = False,
    oracle_reason: str = "",
) -> ReferenceCase:
    blocks = []
    sections = []
    offsets: dict[str, int] = defaultdict(int)
    ordinals: dict[str, int] = defaultdict(int)
    for section, block in entries:
        document_id = block.document_revision_id
        ordinal = ordinals[document_id]
        start = offsets[document_id]
        block = replace(
            block,
            ordinal=ordinal,
            canonical_char_start=start,
            canonical_char_end=start + len(block.text),
        )
        blocks.append(block)
        sections.append(
            replace(
                section,
                ordinal=ordinal,
                first_block_ordinal=ordinal,
                last_block_ordinal=ordinal,
                token_estimate=block.token_estimate,
            )
        )
        offsets[document_id] += len(block.text) + 2
        ordinals[document_id] += 1
    expected = set(required_ids)
    seed_set = set(seed_ids)
    full_cost = packet_cost([block for block in blocks if block.block_id in expected])
    seed_cost = packet_cost([block for block in blocks if block.block_id in seed_set])
    if budget_regime == "ample":
        budget = full_cost + 8
    elif budget_regime == "one_unit_short":
        budget = max(seed_cost, full_cost - 1)
    elif budget_regime == "seed_only":
        budget = seed_cost
    else:
        raise ValueError(f"Unknown budget regime: {budget_regime}")
    return ReferenceCase(
        case_id=case_id,
        category=category,
        mutation=mutation,
        budget_regime=budget_regime,
        chain_hops=chain_hops,
        blocks=tuple(blocks),
        sections=tuple(sections),
        seed_block_ids=seed_ids,
        required_block_ids=required_ids,
        oracle_unresolved=oracle_unresolved,
        oracle_reason=oracle_reason,
        budget=budget,
    )


def _chain_case(hops: int, mutation: str, regime: str, rng: random.Random) -> ReferenceCase:
    case_id = f"chain-{hops}-{mutation}-{regime}"
    document_id = f"{case_id}:policy"
    numbers = list(range(1, hops + 2))
    if mutation == "renumbered":
        numbers = rng.sample(range(20, 900), hops + 1)
    entries = []
    required = []
    for position, number in enumerate(numbers):
        section_id = f"{case_id}:section:{position}"
        block_id = f"{case_id}:block:{position}"
        required.append(block_id)
        if position == hops:
            text = "Contractors are excluded from this eligibility rule."
        elif mutation == "reference_wording":
            text = f"Eligibility is conditional. See Section {numbers[position + 1]} for the exception."
        else:
            text = f"Staff are eligible subject to Section {numbers[position + 1]}."
        entries.append(
            (_section(section_id, document_id, number, "Eligibility"),
             _block(block_id, document_id, section_id, text))
        )
    for position in range(3):
        section_id = f"{case_id}:filler-section:{position}"
        text = ("This unrelated paragraph describes meeting-room furniture. " * (position + 1)).strip()
        entries.append(
            (_section(section_id, document_id, 1001 + position, "Furniture"),
             _block(f"{case_id}:filler:{position}", document_id, section_id, text))
        )
    if mutation == "cross_document_decoy":
        other_document = f"{case_id}:other-policy"
        for position, number in enumerate(numbers):
            section_id = f"{case_id}:decoy-section:{position}"
            entries.append(
                (_section(section_id, other_document, number, "Other policy"),
                 _block(f"{case_id}:decoy:{position}", other_document, section_id,
                        "Contractors are included under this unrelated policy."))
            )
    if mutation == "permuted_filler":
        rng.shuffle(entries)
    return _assemble(
        case_id=case_id,
        category="reference_chain",
        mutation=mutation,
        budget_regime=regime,
        chain_hops=hops,
        entries=entries,
        seed_ids=(required[0],),
        required_ids=tuple(required),
    )


def _special_case(category: str, variant: int, rng: random.Random) -> ReferenceCase:
    case_id = f"{category}-{variant}"
    document_id = f"{case_id}:policy"
    base = rng.randrange(100, 500)
    numbers = [base, base + 1, base + 2, base + 3]
    texts = [
        f"Staff are eligible subject to Section {numbers[1]}.",
        f"Eligibility depends on Section {numbers[2]}.",
        "Contractors are excluded from this eligibility rule.",
    ]
    required_positions = [0, 1, 2]
    oracle_unresolved = category in {"missing_target", "ambiguous_target", "missing_intermediate"}
    reason = ""
    if category == "section_range":
        texts[0] = f"Staff are eligible subject to Sections {numbers[1]}-{numbers[2]}."
        texts[1] = "An employee must complete orientation before becoming eligible."
    elif category == "cycle":
        texts[2] = f"Contractors are excluded. This exception qualifies Section {numbers[0]}."
    elif category == "ambiguous_target":
        numbers[2] = numbers[1]
        texts[1] = "Contractors are excluded under the first competing section."
        texts[2] = "Contractors are eligible under the second competing section."
        reason = "The cited section label names two distinct sections in the same document."
    elif category in {"missing_target", "missing_intermediate"}:
        reason = "The fixture deliberately removes a required referenced section."
    else:
        raise ValueError(f"Unknown category: {category}")
    entries = []
    required_ids = tuple(f"{case_id}:block:{position}" for position in required_positions)
    for position, text in enumerate(texts):
        if category == "missing_target" and position == 2:
            continue
        if category == "missing_intermediate" and position == 1:
            continue
        section_id = f"{case_id}:section:{position}"
        entries.append(
            (_section(section_id, document_id, numbers[position], "Eligibility"),
             _block(required_ids[position], document_id, section_id, text))
        )
    filler_section = f"{case_id}:filler-section"
    entries.append(
        (_section(filler_section, document_id, 1001, "Furniture"),
         _block(f"{case_id}:filler", document_id, filler_section,
                "This unrelated paragraph describes meeting-room furniture."))
    )
    rng.shuffle(entries)
    return _assemble(
        case_id=case_id,
        category=category,
        mutation=f"renumbered_permutation_{variant}",
        budget_regime="ample" if variant % 2 == 0 else "seed_only",
        chain_hops=2,
        entries=entries,
        seed_ids=(required_ids[0],),
        required_ids=required_ids,
        oracle_unresolved=oracle_unresolved,
        oracle_reason=reason,
    )


def _direct_case(variant: int, rng: random.Random) -> ReferenceCase:
    case_id = f"direct-control-{variant}"
    document_id = f"{case_id}:policy"
    section_id = f"{case_id}:section:0"
    block_id = f"{case_id}:block:0"
    direct_texts = (
        "Contractors are excluded from the employee orientation benefit.",
        "Employees may reserve a meeting room for up to two hours.",
        "The orientation session begins at nine in the morning.",
        "The training room is located on the second floor.",
    )
    entries = [
        (_section(section_id, document_id, rng.randrange(1, 900), "Direct information"),
         _block(block_id, document_id, section_id, direct_texts[variant % len(direct_texts)]))
    ]
    for position in range(3):
        filler_section = f"{case_id}:filler-section:{position}"
        entries.append(
            (_section(filler_section, document_id, 1001 + position, "Furniture"),
             _block(f"{case_id}:filler:{position}", document_id, filler_section,
                    "This unrelated paragraph describes meeting-room furniture."))
        )
    rng.shuffle(entries)
    return _assemble(
        case_id=case_id,
        category="direct_no_reference_control",
        mutation=f"direct_wording_permutation_{variant}",
        budget_regime="ample" if variant % 2 == 0 else "seed_only",
        chain_hops=0,
        entries=entries,
        seed_ids=(block_id,),
        required_ids=(block_id,),
    )


def generate_cases(seed: int = DEFAULT_SEED) -> tuple[ReferenceCase, ...]:
    """Construct 80 stress cases and 20 direct controls without user documents."""
    rng = random.Random(seed)
    cases = [
        _chain_case(hops, mutation, regime, rng)
        for hops in (1, 2, 3, 5)
        for mutation in ("canonical", "renumbered", "permuted_filler", "reference_wording", "cross_document_decoy")
        for regime in ("ample", "one_unit_short", "seed_only")
    ]
    cases.extend(
        _special_case(category, variant, rng)
        for category in ("missing_target", "ambiguous_target", "missing_intermediate", "section_range", "cycle")
        for variant in range(4)
    )
    cases.extend(_direct_case(variant, rng) for variant in range(20))
    return tuple(cases)


def _two_hop_blocks(case: ReferenceCase) -> tuple[SourceBlock, ...]:
    """Independent limited-hop baseline for the fixture's explicit grammar.

    It supports numbered section references/ranges and same-document lookup.
    Ambiguous or missing references are silently skipped, as in a naive expander.
    This is a diagnostic baseline, not a reimplementation of a published system.
    """
    by_id = {block.block_id: block for block in case.blocks}
    section_blocks: dict[str, list[SourceBlock]] = defaultdict(list)
    section_labels: dict[tuple[str, str], list[str]] = defaultdict(list)
    for block in case.blocks:
        if block.section_id:
            section_blocks[block.section_id].append(block)
    for section in case.sections:
        match = re.match(r"Section\s+(\d+)\b", section.heading, re.IGNORECASE)
        if match:
            section_labels[(section.document_revision_id, match[1])].append(section.section_id)
    selected = {block_id: by_id[block_id] for block_id in case.seed_block_ids}
    frontier = list(selected.values())
    for _ in range(2):
        next_frontier = []
        for block in frontier:
            for match in re.finditer(r"\bSections?\s+(\d+)(?:\s*-\s*(\d+))?", block.text, re.IGNORECASE):
                first = int(match[1])
                last = int(match[2]) if match[2] else first
                for number in range(first, last + 1):
                    candidates = section_labels.get((block.document_revision_id, str(number)), [])
                    if len(candidates) != 1:
                        continue
                    for target in section_blocks[candidates[0]]:
                        if target.block_id in selected:
                            continue
                        if packet_cost((*selected.values(), target)) > case.budget:
                            continue
                        selected[target.block_id] = target
                        next_frontier.append(target)
        frontier = next_frontier
    return tuple(selected.values())


def _score(case: ReferenceCase, method: str, blocks: Sequence[SourceBlock], issues: Sequence[str]) -> dict[str, Any]:
    presented_ids = tuple(dict.fromkeys(block.block_id for block in blocks))
    presented = set(presented_ids)
    required = set(case.required_block_ids)
    missing = required - presented
    dependency_complete = not missing
    context_complete = dependency_complete and not case.oracle_unresolved
    flagged = bool(issues)
    required_available = tuple(block for block in case.blocks if block.block_id in required)
    oracle_budget_feasible = (
        not case.oracle_unresolved
        and {block.block_id for block in required_available} == required
        and packet_cost(required_available) <= case.budget
    )
    return {
        "case_id": case.case_id,
        "method": method,
        "category": case.category,
        "mutation": case.mutation,
        "chain_hops": case.chain_hops,
        "budget_regime": case.budget_regime,
        "budget": case.budget,
        "required_block_ids": list(case.required_block_ids),
        "presented_block_ids": list(presented_ids),
        "missing_block_ids": sorted(missing),
        "required_blocks": len(required),
        "present_required_blocks": len(required & presented),
        "extra_blocks": len(presented - required),
        "dependency_complete": dependency_complete,
        "oracle_unresolved": case.oracle_unresolved,
        "context_complete": context_complete,
        "oracle_budget_feasible": oracle_budget_feasible,
        "flagged": flagged,
        "declares_complete": not flagged,
        "useful_completed_context": context_complete and not flagged,
        "issues": list(dict.fromkeys(issues)),
        "unflagged_incomplete_release": not context_complete and not flagged,
        "flagged_complete_context": context_complete and flagged,
        "token_cost": packet_cost(blocks),
        "over_budget": packet_cost(blocks) > case.budget,
    }


def evaluate_case(case: ReferenceCase) -> tuple[dict[str, Any], ...]:
    by_id = {block.block_id: block for block in case.blocks}
    seed_blocks = tuple(by_id[block_id] for block_id in case.seed_block_ids)
    seed_revisions = {block.document_revision_id for block in seed_blocks}
    if len(seed_revisions) != 1:
        raise ValueError("Each benchmark case must select seeds from one revision")
    # Production resolves references inside a single selected revision. Decoy
    # documents remain in the fixture and scorer but cannot enter this index.
    index = ReferenceIndex(
        tuple(block for block in case.blocks if block.document_revision_id in seed_revisions),
        tuple(section for section in case.sections if section.document_revision_id in seed_revisions),
    )
    packets = pack_closed(index, seed_blocks, case.budget, packet_cost)
    packed = tuple({block.block_id: block for packet in packets for block in packet.blocks}.values())
    packing_issues = tuple(issue for packet in packets for issue in packet.issues)
    delivery_audit = index.audit(case.seed_block_ids, tuple(block.block_id for block in packed))
    # Ablation deliberately simulates an intermediate optimizer retaining only
    # seed evidence after successful expansion. Only the post-delivery audit is
    # removed; earlier packing warnings remain observable.
    stripped = tuple(block for block in packed if block.block_id in case.seed_block_ids)
    return (
        _score(case, METHODS[0], seed_blocks, ()),
        _score(case, METHODS[1], _two_hop_blocks(case), ()),
        _score(case, METHODS[2], packed, (*packing_issues, *delivery_audit.issues)),
        _score(case, METHODS[3], stripped, packing_issues),
        _score(case, METHODS[4], (), ("baseline_abstention:always",)),
    )


def _fraction(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    incomplete = sum(not row["context_complete"] for row in rows)
    feasible = [row for row in rows if row["oracle_budget_feasible"]]
    return {
        "cases": count,
        "dependency_block_recall": _fraction(sum(row["present_required_blocks"] for row in rows), sum(row["required_blocks"] for row in rows)),
        "dependency_complete_cases": _fraction(sum(row["dependency_complete"] for row in rows), count),
        "context_complete_cases": _fraction(sum(row["context_complete"] for row in rows), count),
        "declared_complete_cases": _fraction(sum(row["declares_complete"] for row in rows), count),
        "useful_completed_context": _fraction(sum(row["useful_completed_context"] for row in rows), count),
        "useful_completion_given_feasible": _fraction(sum(row["useful_completed_context"] for row in feasible), len(feasible)),
        "unflagged_incomplete_releases": _fraction(sum(row["unflagged_incomplete_release"] for row in rows), count),
        "unflagged_given_incomplete": _fraction(sum(row["unflagged_incomplete_release"] for row in rows), incomplete),
        "flagged_cases": _fraction(sum(row["flagged"] for row in rows), count),
        "flagged_complete_context": _fraction(sum(row["flagged_complete_context"] for row in rows), sum(row["context_complete"] for row in rows)),
        "over_budget_cases": _fraction(sum(row["over_budget"] for row in rows), count),
        "extra_blocks_total": sum(row["extra_blocks"] for row in rows),
        "token_cost_total": sum(row["token_cost"] for row in rows),
        "token_cost_mean": sum(row["token_cost"] for row in rows) / count if count else None,
    }


def run_benchmark(seed: int = DEFAULT_SEED) -> dict[str, Any]:
    cases = generate_cases(seed)
    fixture_json = json.dumps([asdict(case) for case in cases], sort_keys=True, ensure_ascii=False)
    rows = [row for case in cases for row in evaluate_case(case)]
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "fixtures_sha256": hashlib.sha256(fixture_json.encode("utf-8")).hexdigest(),
        "scope": "Synthetic structural context preservation; no LLM calls, answer-quality judgments, real-corpus results, or state-of-the-art claims.",
        "cost_definition": "sum(ceil(len(block.text)/4) + 8) over presented blocks; surrogate token units, not a model tokenizer",
        "oracle_definition": "Fixture-authored dependency IDs plus deliberately missing/ambiguous reference labels; independent of ReferenceIndex output.",
        "release_definition": "An unflagged packet is treated as releasable. Flagged packets must be routed to partial/abstain by an integrating application; this benchmark does not execute that application policy.",
        "utility_definition": "Useful completed context means all oracle dependencies delivered, no unresolved oracle reference, and no issue flag. This is structural utility, not measured answer usefulness. Feasibility additionally requires all oracle dependencies to fit the budget.",
        "cases_total": len(cases),
        "oracle_unresolved_cases": sum(case.oracle_unresolved for case in cases),
        "methods": {method: _aggregate([row for row in rows if row["method"] == method]) for method in METHODS},
        "by_category": {
            category: {method: _aggregate([row for row in rows if row["method"] == method and row["category"] == category]) for method in METHODS}
            for category in sorted({case.category for case in cases})
        },
        "by_budget_regime": {
            regime: {method: _aggregate([row for row in rows if row["method"] == method and row["budget_regime"] == regime]) for method in METHODS}
            for regime in sorted({case.budget_regime for case in cases})
        },
        "cases": rows,
    }


def _display_fraction(value: dict[str, Any]) -> str:
    return f"{value['numerator']}/{value['denominator']}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Reference-context robustness: synthetic functional benchmark",
        "",
        f"Seed: `{report['seed']}`. Cases: **{report['cases_total']}**. Fixture SHA-256: `{report['fixtures_sha256']}`.",
        "",
        report["scope"],
        "",
        "The benchmark contains 80 reference stress cases and 20 direct no-reference controls. Reference cases start with an eligibility clause and explicit references leading to a contractor exclusion. The oracle records required block IDs during fixture construction. Filler permutations, renumbering, different reference wording, competing documents, ranges, cycles, missing sections and ambiguous sections exercise structural failure modes.",
        "",
        "| Method | Complete dependencies | Useful complete / all | Useful / feasible | Unflagged incomplete / all | Unflagged / incomplete | Over budget | Extra blocks | Mean cost |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, summary in report["methods"].items():
        lines.append(
            f"| {method} | {_display_fraction(summary['dependency_complete_cases'])} | {_display_fraction(summary['useful_completed_context'])} | {_display_fraction(summary['useful_completion_given_feasible'])} | {_display_fraction(summary['unflagged_incomplete_releases'])} | {_display_fraction(summary['unflagged_given_incomplete'])} | {_display_fraction(summary['over_budget_cases'])} | {summary['extra_blocks_total']} | {summary['token_cost_mean']:.2f} |"
        )
    lines.extend([
        "",
        "## Interpretation and limits",
        "",
        "- Dependency completeness scores the actual delivered block IDs, including the seed. Unresolvable fixtures additionally require a flag even if all existing candidates were delivered.",
        f"- Utility: {report['utility_definition']}",
        "- The always-abstain control delivers no blocks and flags every case: zero unflagged incomplete packets, but also zero useful completions. The 20 direct controls check whether ordinary complete context remains available.",
        f"- {report['oracle_unresolved_cases']} fixtures deliberately contain missing or ambiguous targets. A flag is the intended behavior there; retrieving a missing block is impossible.",
        "- Original selection retains the seed. The two-hop baseline independently follows explicit same-document section references and greedily fits the same budget. Neither baseline flags truncation or unresolved targets.",
        "- The caller scopes the transitive index to the seed document revision, as required by its API. Cross-document decoys exercise this caller scoping; they do not measure automatic document disambiguation.",
        "- Transitive closure audits the delivered IDs. The synthesis-stripping ablation removes added context after packing and omits that final audit, while preserving packing warnings. This is an intentional destructive transformation, not measured LLM behavior.",
        f"- Cost: {report['cost_definition']}.",
        f"- Release: {report['release_definition']}",
        "- Cases are controlled mutations, not independent samples from a population. No statistical significance, semantic entailment, published-system superiority, publication readiness or state-of-the-art performance is claimed.",
        "- A real evaluation still needs natural documents, human reference annotations, downstream answer judgments, stronger retrieval baselines and an unseen held-out set.",
        "",
        "## Reproduction",
        "",
        f"Run from the research project directory: `python -m sisu_reader.reference_evaluation --seed {report['seed']} --output reference_benchmark_results`.",
        "",
        "`reference_benchmark.json` contains all case outcomes and aggregate numerators/denominators. `reference_fixtures.json` contains the full synthetic fixtures and oracle annotations. No private documents or model credentials are read.",
        "",
    ])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, required=True, help="Directory for synthetic fixtures, JSON results and Markdown report")
    args = parser.parse_args(argv)
    report = run_benchmark(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "reference_benchmark.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output / "reference_benchmark.md").write_text(render_markdown(report), encoding="utf-8")
    (args.output / "reference_fixtures.json").write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "seed": args.seed, "cases": [asdict(case) for case in generate_cases(args.seed)]}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {report['cases_total']} synthetic cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
