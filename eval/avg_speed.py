#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

RE_MODEL = re.compile(r"Evaluating\s+([^:]+):")
RE_SPEED = re.compile(r"([0-9]*\.?[0-9]+)\s*(image/s|images/s|s/image)")


@dataclass
class SpeedSums:
    count: int = 0
    sum_ips: float = 0.0


def parse_speed_to_ips(value: float, unit: str) -> Optional[float]:
    if unit in ("image/s", "images/s"):
        return value
    if unit == "s/image":
        if value == 0:
            return None
        return 1.0 / value
    return None


def iter_speed_records(lines: Iterable[str]) -> Iterable[tuple[str, float]]:
    for line in lines:
        model_match = RE_MODEL.search(line)
        speed_match = RE_SPEED.search(line)
        if not (model_match and speed_match):
            continue

        model = model_match.group(1).strip()
        value = float(speed_match.group(1))
        unit = speed_match.group(2)
        ips = parse_speed_to_ips(value, unit)
        if ips is None:
            continue
        yield model, ips


def aggregate_speeds(lines: Iterable[str]) -> Dict[str, SpeedSums]:
    totals: Dict[str, SpeedSums] = {}
    for model, ips in iter_speed_records(lines):
        if model not in totals:
            totals[model] = SpeedSums()
        totals[model].count += 1
        totals[model].sum_ips += ips
    return totals


def compute_averages(totals: Dict[str, SpeedSums]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model, entry in totals.items():
        if entry.count == 0:
            continue
        rows.append(
            {
                "model": model,
                "count": entry.count,
                "avg_images_per_second": entry.sum_ips / entry.count,
            }
        )
    rows.sort(key=lambda r: r["model"])
    return rows


def write_table(rows: List[Dict[str, object]], output) -> None:
    headers = ["model", "count", "avg_images_per_second"]
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
    fieldnames = ["model", "count", "avg_images_per_second"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


def write_json(rows: List[Dict[str, object]], output) -> None:
    json.dump(rows, output, indent=2, sort_keys=False)
    output.write("\n")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate average images/sec per model from .err logs.")
    parser.add_argument("input", help="Path to the .err log file.")
    parser.add_argument("--format", choices=["table", "csv", "json"], default="table", help="Output format.")
    parser.add_argument("--output", default="-", help="Output path or '-' for stdout (default).")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    with open(args.input, "r", encoding="utf-8") as handle:
        totals = aggregate_speeds(handle)
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

