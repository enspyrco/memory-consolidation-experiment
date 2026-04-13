#!/usr/bin/env python3
"""
Analyze experiment results and generate comparison tables.

Usage:
  python analyze_results.py results/
  python analyze_results.py results/ --format markdown
"""

import json
import sys
from pathlib import Path


CATEGORIES = {
    1: "Multi-hop",
    2: "Single-hop",
    3: "Temporal",
    4: "Open-domain",
    5: "Adversarial",
}


def load_results(results_dir: str) -> dict:
    """Load all condition result files from a directory."""
    results = {}
    for path in sorted(Path(results_dir).glob("*.json")):
        if path.name == "combined.json":
            continue
        try:
            data = json.loads(path.read_text())
            if "condition" in data:
                results[data["condition"]] = data
        except Exception as e:
            print(f"  Skipping {path.name}: {e}")
    return results


def print_summary(results: dict, fmt: str = "text"):
    """Print a summary comparison table."""
    if not results:
        print("No results found.")
        return

    if fmt == "markdown":
        print("| Condition | F1 | Multi-hop F1 | Tokens | Skip % |")
        print("|---|---:|---:|---:|---:|")
        for name, data in results.items():
            overall = data["aggregate"].get("overall", {})
            cat1 = data["aggregate"].get("category_1", {})
            f1 = overall.get("f1", {}).get("mean", 0)
            mh_f1 = cat1.get("f1", {}).get("mean", 0)
            tokens = data["total_tokens"]
            skip = data["routing"]["skip_pct"]
            print(f"| {name} | {f1:.3f} | {mh_f1:.3f} | {tokens:,} | {skip:.1f}% |")
    else:
        print(f"\n{'Condition':<22} {'F1':>6} {'Multi-hop F1':>13} {'Tokens':>10} {'Skip %':>8}")
        print("-" * 65)
        for name, data in results.items():
            overall = data["aggregate"].get("overall", {})
            cat1 = data["aggregate"].get("category_1", {})
            f1 = overall.get("f1", {}).get("mean", 0)
            mh_f1 = cat1.get("f1", {}).get("mean", 0)
            tokens = data["total_tokens"]
            skip = data["routing"]["skip_pct"]
            print(f"{name:<22} {f1:>6.3f} {mh_f1:>13.3f} {tokens:>10,} {skip:>7.1f}%")

    # Per-category breakdown
    print("\n\nPer-category F1:")
    if fmt == "markdown":
        header = "| Condition |" + "|".join(f" {v} " for v in CATEGORIES.values()) + "|"
        sep = "|---|" + "|".join("---:" for _ in CATEGORIES) + "|"
        print(header)
        print(sep)
        for name, data in results.items():
            cells = []
            for cat_id in CATEGORIES:
                cat_data = data["aggregate"].get(f"category_{cat_id}", {})
                f1 = cat_data.get("f1", {}).get("mean", 0)
                cells.append(f"{f1:.3f}")
            print(f"| {name} | " + " | ".join(cells) + " |")
    else:
        print(f"{'Condition':<22}" + "".join(f"{v:>12}" for v in CATEGORIES.values()))
        print("-" * (22 + 12 * len(CATEGORIES)))
        for name, data in results.items():
            cells = []
            for cat_id in CATEGORIES:
                cat_data = data["aggregate"].get(f"category_{cat_id}", {})
                f1 = cat_data.get("f1", {}).get("mean", 0)
                cells.append(f"{f1:>12.3f}")
            print(f"{name:<22}" + "".join(cells))

    # Routing breakdown
    print("\n\nRouting breakdown:")
    if fmt == "markdown":
        print("| Condition | Total Turns | Skip | Construct | Evolve |")
        print("|---|---:|---:|---:|---:|")
        for name, data in results.items():
            r = data["routing"]
            print(f"| {name} | {r['total_turns']} | {r['skip_pct']:.1f}% | {r['construct_pct']:.1f}% | {r['evolve_pct']:.1f}% |")
    else:
        print(f"{'Condition':<22} {'Turns':>8} {'Skip':>8} {'Construct':>10} {'Evolve':>8}")
        print("-" * 60)
        for name, data in results.items():
            r = data["routing"]
            print(
                f"{name:<22} {r['total_turns']:>8} "
                f"{r['skip_pct']:>7.1f}% {r['construct_pct']:>9.1f}% {r['evolve_pct']:>7.1f}%"
            )


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results_dir> [--format markdown]")
        sys.exit(1)

    results_dir = sys.argv[1]
    fmt = "markdown" if "--format" in sys.argv and "markdown" in sys.argv else "text"

    results = load_results(results_dir)
    print(f"Loaded {len(results)} conditions from {results_dir}")
    print_summary(results, fmt)


if __name__ == "__main__":
    main()
