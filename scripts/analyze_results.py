from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from rag_forecast.metrics import spearman  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a per-question results CSV.")
    parser.add_argument("csv", type=Path, help="Path to per-question results CSV.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write summary.json (default: alongside the input CSV).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    n = len(df)
    rho, p = spearman(df["abs_z"].tolist(), df["brier_delta"].tolist())

    summary = {
        "n": n,
        "mean_brier_h": float(df["brier_h"].mean()) if n else None,
        "mean_brier_he": float(df["brier_he"].mean()) if n else None,
        "mean_brier_delta": float(df["brier_delta"].mean()) if n else None,
        "frac_brier_improved": float((df["brier_delta"] > 0).mean()) if n else None,
        "mean_abs_z": float(df["abs_z"].mean()) if n else None,
        "frac_z_positive": float((df["z"] > 0).mean()) if n else None,
        "spearman_abs_z_vs_brier_delta": {"rho": rho, "p_value": p},
    }

    out = args.out or args.csv.with_name(args.csv.stem + "_summary.json")
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
