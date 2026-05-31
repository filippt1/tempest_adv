#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

RE_MODEL = re.compile(r"\bmodel=([^\s]+)")
RE_NAME = re.compile(r"\bname=([^\s|]+)")
RE_METRICS_BLOCK = re.compile(r"restored\[([^\]]+)\]")
RE_METRIC_PAIR = re.compile(r"([A-Za-z0-9_]+)=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

REQUIRED_METRICS = ("mse", "psnr", "ssim", "3ssim")


@dataclass
class MetricSums:
    count: int = 0
    sums: Dict[str, float] = None

    def __post_init__(self) -> None:
        if self.sums is None:
            self.sums = {k: 0.0 for k in REQUIRED_METRICS}


def parse_metrics_block(block: str) -> Dict[str, float]:
    pairs: Dict[str, float] = {}
    for match in RE_METRIC_PAIR.finditer(block):
        key = match.group(1)
        value = float(match.group(2))
        pairs[key] = value
    return pairs


def iter_matching_records(lines: Iterable[str], name_prefix: str, case_insensitive: bool) -> Iterable[Tuple[str, Dict[str, float]]]:
    if case_insensitive:
        name_prefix_cmp = name_prefix.lower()
    else:
        name_prefix_cmp = name_prefix

    for line in lines:
        model_match = RE_MODEL.search(line)
        name_match = RE_NAME.search(line)
        metrics_match = RE_METRICS_BLOCK.search(line)
        if not (model_match and name_match and metrics_match):
            continue

        name_value = name_match.group(1)
        name_cmp = name_value.lower() if case_insensitive else name_value
        if not name_cmp.startswith(name_prefix_cmp):
            continue

        metrics = parse_metrics_block(metrics_match.group(1))
        if not all(k in metrics for k in REQUIRED_METRICS):
            continue

        yield model_match.group(1), metrics


def aggregate_metrics(lines: Iterable[str], name_prefix: str, case_insensitive: bool) -> Dict[str, MetricSums]:
    totals: Dict[str, MetricSums] = {}
    for model, metrics in iter_matching_records(lines, name_prefix, case_insensitive):
        if model not in totals:
            totals[model] = MetricSums()
        entry = totals[model]
        entry.count += 1
        for key in REQUIRED_METRICS:
            entry.sums[key] += metrics[key]
    return totals


def compute_averages(totals: Dict[str, MetricSums]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model, entry in totals.items():
        if entry.count == 0:
            continue
        row = {"model": model, "count": entry.count}
        for key in REQUIRED_METRICS:
            row[f"avg_{key}"] = entry.sums[key] / entry.count
        rows.append(row)
    rows.sort(key=lambda r: r["model"])
    return rows


def write_table(rows: List[Dict[str, object]], output) -> None:
    headers = ["model", "count", "avg_mse", "avg_psnr", "avg_ssim", "avg_3ssim"]
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            val = row[h]
            if isinstance(val, float):
                text = f"{val:.4f}"
            else:
                text = str(val)
            widths[h] = max(widths[h], len(text))

    header_line = "  ".join(h.ljust(widths[h]) for h in headers)
    output.write(header_line + "\n")
    output.write("  ".join("-" * widths[h] for h in headers) + "\n")

    for row in rows:
        parts = []
        for h in headers:
            val = row[h]
            if isinstance(val, float):
                text = f"{val:.4f}"
            else:
                text = str(val)
            parts.append(text.ljust(widths[h]))
        output.write("  ".join(parts) + "\n")


def write_csv(rows: List[Dict[str, object]], output) -> None:
    fieldnames = ["model", "count", "avg_mse", "avg_psnr", "avg_ssim", "avg_3ssim"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


def write_json(rows: List[Dict[str, object]], output) -> None:
    json.dump(rows, output, indent=2, sort_keys=False)
    output.write("\n")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate mse/psnr/ssim/3ssim averages per model for animal* names.")
    parser.add_argument("input", help="Path to the .out log file.")
    parser.add_argument("--name-prefix", default="animal", help="Name prefix to match (default: animal).")
    parser.add_argument("--case-insensitive", action="store_true", help="Match names case-insensitively.")
    parser.add_argument("--format", choices=["table", "csv", "json"], default="table", help="Output format.")
    parser.add_argument("--output", default="-", help="Output path or '-' for stdout (default).")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    with open(args.input, "r", encoding="utf-8") as handle:
        totals = aggregate_metrics(handle, args.name_prefix, args.case_insensitive)
    rows = compute_averages(totals)

    output = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8", newline="")
    try:
        if args.format == "table":
            write_table(rows, output)
        elif args.format == "csv":
            write_csv(rows, output)
        else:
            write_json(rows, output)
    finally:
        if output is not sys.stdout:
            output.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

