#!/usr/bin/env python3
"""Compute character error rate (CER) for restored images vs. ground truth."""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
import fastwer
import pytesseract


def parse_restored_args(restored_args: List[str]) -> List[Tuple[str, Path]]:
    restored: List[Tuple[str, Path]] = []
    for item in restored_args:
        if "=" not in item:
            raise ValueError(f"Invalid --restored format: {item}. Use NAME=PATH.")
        name, path = item.split("=", 1)
        if not name:
            raise ValueError(f"Empty model name in --restored: {item}")
        restored.append((name, Path(path)))
    return restored


def extract_text(image_path: Path) -> str:
    with Image.open(image_path) as img:
        text = pytesseract.image_to_string(img)
    return text.strip().replace("\n", " ")


def cer_for_pair(restored_path: Path, gt_path: Path) -> float | None:
    est_text = extract_text(restored_path)
    gt_text = extract_text(gt_path)
    if not gt_text:
        return None
    # fastwer.score returns percentage; convert to [0, 1] range.
    score = fastwer.score([est_text], [gt_text], char_level=True) / 100.0
    if score != score:  # NaN guard
        return None
    return score


def compute_model_cer(restored_dir: Path, gt_dir: Path) -> Tuple[float, int, int, List[Tuple[str, str, float]]]:
    scores: List[float] = []
    skipped = 0
    per_pair: List[Tuple[str, str, float]] = []
    for item in sorted(restored_dir.iterdir()):
        if item.name.startswith("."):
            continue
        if not item.is_file():
            continue
        if len(item.name) <= 7:
            skipped += 1
            continue
        gt_name = item.name[7:]
        gt_path = gt_dir / gt_name
        if not gt_path.exists():
            skipped += 1
            continue
        score = cer_for_pair(item, gt_path)
        if score is None:
            skipped += 1
            continue
        scores.append(score)
        per_pair.append((item.name, gt_name, score))
        print(f"pair {item.name} vs {gt_name}: CER={score:.6f}")
    avg = sum(scores) / len(scores) if scores else float("nan")
    return avg, len(scores), skipped, per_pair


def compute_baseline_cer(baseline_dir: Path) -> Tuple[float, int, int, List[Tuple[str, str, float]]]:
    scores: List[float] = []
    skipped = 0
    per_pair: List[Tuple[str, str, float]] = []
    for gt_item in sorted(baseline_dir.iterdir()):
        if gt_item.name.startswith("."):
            continue
        if not gt_item.is_file():
            continue
        noisy_dir = baseline_dir / gt_item.stem
        if not noisy_dir.exists() or not noisy_dir.is_dir():
            skipped += 1
            continue
        for noisy_item in sorted(noisy_dir.iterdir()):
            if noisy_item.name.startswith("."):
                continue
            if not noisy_item.is_file():
                continue
            score = cer_for_pair(noisy_item, gt_item)
            if score is None:
                skipped += 1
                continue
            scores.append(score)
            per_pair.append((noisy_item.name, gt_item.name, score))
            print(f"pair {noisy_item.name} vs {gt_item.name}: CER={score:.6f}")
    avg = sum(scores) / len(scores) if scores else float("nan")
    return avg, len(scores), skipped, per_pair


def write_csv(rows: List[Tuple[str, float, int, int]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "cer_avg", "pairs_used", "pairs_skipped"])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute CER for restored images against ground truth images."
    )
    parser.add_argument(
        "--restored",
        nargs="+",
        required=False,
        help="One or more NAME=PATH pairs for restored images.",
    )
    parser.add_argument(
        "--gt-dir",
        required=False,
        help="Path to ground truth directory (e.g., test).")
    parser.add_argument(
        "--baseline-dir",
        help=(
            "Path to baseline dataset root (contains GT images and matching noisy-image folders)."
        ),
    )
    parser.add_argument(
        "--baseline-name",
        default="baseline",
        help="Name to use for baseline results in the CSV.")
    parser.add_argument(
        "--out-csv",
        default="cer_results.csv",
        help="Path to write average CER results as CSV.")
    args = parser.parse_args()

    if not args.restored and not args.baseline_dir:
        raise ValueError("Provide --restored and/or --baseline-dir.")

    restored_pairs: List[Tuple[str, Path]] = []
    gt_dir: Path | None = None
    if args.restored:
        restored_pairs = parse_restored_args(args.restored)
        if not args.gt_dir:
            raise ValueError("--gt-dir is required when using --restored.")
        gt_dir = Path(args.gt_dir)
        if not gt_dir.exists():
            raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")

    results: Dict[str, Tuple[float, int, int]] = {}
    csv_rows: List[Tuple[str, float, int, int]] = []
    for name, restored_dir in restored_pairs:
        if not restored_dir.exists():
            raise FileNotFoundError(f"Restored directory not found: {restored_dir}")
        avg, used, skipped, _ = compute_model_cer(restored_dir, gt_dir)
        results[name] = (avg, used, skipped)
        csv_rows.append((name, avg, used, skipped))

    if args.baseline_dir:
        baseline_root = Path(args.baseline_dir)
        if not baseline_root.exists():
            raise FileNotFoundError(f"Baseline directory not found: {baseline_root}")
        avg, used, skipped, _ = compute_baseline_cer(baseline_root)
        results[args.baseline_name] = (avg, used, skipped)
        csv_rows.append((args.baseline_name, avg, used, skipped))

    for name, (avg, used, skipped) in results.items():
        print(f"{name}: CER={avg:.6f} over {used} pairs (skipped {skipped})")

    write_csv(csv_rows, Path(args.out_csv))


if __name__ == "__main__":
    main()

