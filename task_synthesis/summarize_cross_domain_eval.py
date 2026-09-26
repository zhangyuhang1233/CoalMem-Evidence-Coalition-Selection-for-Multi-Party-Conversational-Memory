#!/usr/bin/env python3
"""Aggregate GroupMemBench accuracy tables across domains.

The input for each domain is ``<results-root>/<domain>/accuracy.tsv``.  Each
question-type cell stores an exact ``correct/total`` count.  Counts are summed
before percentages are computed, so the Total row is a micro-average rather
than an average of domain-level percentages.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_DOMAINS = ("Finance", "Technology", "Healthcare", "Manufacturing")
DEFAULT_QTYPES = (
    "multi_hop",
    "knowledge_update",
    "temporal",
    "user_implicit",
    "term_ambiguity",
    "abstention",
)
COUNT_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")
Count = Tuple[int, int]


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_tsv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"missing accuracy file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"accuracy file has no data rows: {path}")
    if "baseline" not in rows[0]:
        raise ValueError(f"accuracy file has no 'baseline' column: {path}")
    return rows


def choose_baseline(
    rows_by_domain: Mapping[str, Sequence[Mapping[str, str]]],
    requested: Optional[str],
) -> str:
    available = {
        domain: {str(row.get("baseline", "")).strip() for row in rows}
        for domain, rows in rows_by_domain.items()
    }
    for domain, names in available.items():
        names.discard("")
        if not names:
            raise ValueError(f"no baseline rows found for {domain}")

    if requested:
        missing = [domain for domain, names in available.items() if requested not in names]
        if missing:
            raise ValueError(
                f"baseline {requested!r} is missing from domain(s): {', '.join(missing)}"
            )
        return requested

    common = set.intersection(*(set(names) for names in available.values()))
    if len(common) == 1:
        return next(iter(common))
    if not common:
        details = "; ".join(
            f"{domain}={sorted(names)}" for domain, names in available.items()
        )
        raise ValueError(f"domains have no common baseline; pass --baseline ({details})")
    raise ValueError(
        "multiple common baselines found; pass --baseline: " + ", ".join(sorted(common))
    )


def parse_count(value: str, *, domain: str, qtype: str) -> Count:
    match = COUNT_RE.fullmatch(value or "")
    if not match:
        raise ValueError(f"invalid {domain}/{qtype} count {value!r}; expected correct/total")
    correct, total = (int(match.group(1)), int(match.group(2)))
    if total <= 0:
        raise ValueError(f"non-positive total for {domain}/{qtype}: {correct}/{total}")
    if correct > total:
        raise ValueError(f"correct exceeds total for {domain}/{qtype}: {correct}/{total}")
    return correct, total


def load_counts(
    rows_by_domain: Mapping[str, Sequence[Mapping[str, str]]],
    baseline: str,
    qtypes: Sequence[str],
) -> Dict[str, Dict[str, Count]]:
    result: Dict[str, Dict[str, Count]] = {}
    for domain, rows in rows_by_domain.items():
        matches = [row for row in rows if str(row.get("baseline", "")).strip() == baseline]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one row for baseline {baseline!r} in {domain}, "
                f"found {len(matches)}"
            )
        row = matches[0]
        counts = {
            qtype: parse_count(str(row.get(qtype, "")), domain=domain, qtype=qtype)
            for qtype in qtypes
        }

        summed_correct = sum(correct for correct, _ in counts.values())
        summed_total = sum(total for _, total in counts.values())
        try:
            reported_correct = int(str(row.get("overall_correct", "")).strip())
            reported_total = int(str(row.get("overall_total", "")).strip())
        except ValueError as exc:
            raise ValueError(f"invalid overall counts in {domain}/accuracy.tsv") from exc
        if (summed_correct, summed_total) != (reported_correct, reported_total):
            raise ValueError(
                f"{domain} overall mismatch: qtype sum={summed_correct}/{summed_total}, "
                f"reported={reported_correct}/{reported_total}"
            )
        result[domain] = counts
    return result


def sum_counts(
    counts_by_domain: Mapping[str, Mapping[str, Count]], qtypes: Sequence[str]
) -> Dict[str, Count]:
    return {
        qtype: (
            sum(counts[qtype][0] for counts in counts_by_domain.values()),
            sum(counts[qtype][1] for counts in counts_by_domain.values()),
        )
        for qtype in qtypes
    }


def overall(counts: Mapping[str, Count], qtypes: Sequence[str]) -> Count:
    return (
        sum(counts[qtype][0] for qtype in qtypes),
        sum(counts[qtype][1] for qtype in qtypes),
    )


def format_score(count: Count, precision: int) -> str:
    correct, total = count
    return f"{correct / total * 100:.{precision}f}% ({correct}/{total})"


def markdown_table(
    counts_by_domain: Mapping[str, Mapping[str, Count]],
    total_counts: Mapping[str, Count],
    domains: Sequence[str],
    qtypes: Sequence[str],
    precision: int,
    total_only: bool,
) -> str:
    header = "| domain | " + " | ".join(qtypes) + " | overall |"
    separator = "|" + "---|" * (len(qtypes) + 2)
    lines = [header, separator]
    if not total_only:
        for domain in domains:
            counts = counts_by_domain[domain]
            cells = [format_score(counts[qtype], precision) for qtype in qtypes]
            cells.append(format_score(overall(counts, qtypes), precision))
            lines.append("| " + " | ".join([domain] + cells) + " |")

    total_cells = [format_score(total_counts[qtype], precision) for qtype in qtypes]
    total_cells.append(f"**{format_score(overall(total_counts, qtypes), precision)}**")
    lines.append("| " + " | ".join(["Total"] + total_cells) + " |")
    return "\n".join(lines)


def tsv_table(
    counts_by_domain: Mapping[str, Mapping[str, Count]],
    total_counts: Mapping[str, Count],
    domains: Sequence[str],
    qtypes: Sequence[str],
    total_only: bool,
) -> str:
    header = ["domain"] + list(qtypes) + [
        "overall_correct",
        "overall_total",
        "overall_accuracy",
    ]
    lines = ["\t".join(header)]

    scopes: Iterable[Tuple[str, Mapping[str, Count]]]
    if total_only:
        scopes = (("Total", total_counts),)
    else:
        scopes = [*( (domain, counts_by_domain[domain]) for domain in domains ), ("Total", total_counts)]

    for scope, counts in scopes:
        total_correct, total_total = overall(counts, qtypes)
        row = [scope]
        row.extend(f"{counts[qtype][0]}/{counts[qtype][1]}" for qtype in qtypes)
        row.extend(
            [str(total_correct), str(total_total), f"{total_correct / total_total:.4f}"]
        )
        lines.append("\t".join(row))
    return "\n".join(lines)


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(content + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Micro-average GroupMemBench accuracy across four domains."
    )
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument(
        "--baseline",
        default=None,
        help="Baseline row to aggregate; inferred when exactly one is common to all domains.",
    )
    parser.add_argument(
        "--domains",
        default=",".join(DEFAULT_DOMAINS),
        help="Comma-separated domain names.",
    )
    parser.add_argument(
        "--question-types",
        default=",".join(DEFAULT_QTYPES),
        help="Comma-separated question types in display order.",
    )
    parser.add_argument("--precision", type=int, default=2)
    parser.add_argument(
        "--total-only",
        action="store_true",
        help="Write only the aggregate Total row instead of domain rows plus Total.",
    )
    parser.add_argument(
        "--out-markdown",
        type=Path,
        default=None,
        help="Output path (default: <results-root>/accuracy_all_domains.md).",
    )
    parser.add_argument(
        "--out-tsv",
        type=Path,
        default=None,
        help="Output path (default: <results-root>/accuracy_all_domains.tsv).",
    )
    args = parser.parse_args()

    if args.precision < 0:
        parser.error("--precision must be non-negative")
    results_root = args.results_root.expanduser().resolve()
    if not results_root.is_dir():
        parser.error(f"results root not found: {results_root}")

    domains = parse_csv_list(args.domains)
    qtypes = parse_csv_list(args.question_types)
    if not domains:
        parser.error("--domains must not be empty")
    if not qtypes:
        parser.error("--question-types must not be empty")

    try:
        rows_by_domain = {
            domain: read_tsv(results_root / domain / "accuracy.tsv") for domain in domains
        }
        baseline = choose_baseline(rows_by_domain, args.baseline)
        counts_by_domain = load_counts(rows_by_domain, baseline, qtypes)
        total_counts = sum_counts(counts_by_domain, qtypes)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    markdown = markdown_table(
        counts_by_domain,
        total_counts,
        domains,
        qtypes,
        args.precision,
        args.total_only,
    )
    tsv = tsv_table(
        counts_by_domain, total_counts, domains, qtypes, args.total_only
    )
    out_markdown = args.out_markdown or results_root / "accuracy_all_domains.md"
    out_tsv = args.out_tsv or results_root / "accuracy_all_domains.tsv"
    atomic_write(out_markdown, markdown)
    atomic_write(out_tsv, tsv)

    print(f"baseline: {baseline}")
    print(markdown)
    print()
    print(f"wrote markdown -> {out_markdown}")
    print(f"wrote tsv -> {out_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
