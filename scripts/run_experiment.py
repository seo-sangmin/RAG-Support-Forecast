from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from rag_forecast.config import Config  # noqa: E402
from rag_forecast.pipeline import run  # noqa: E402


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description="Run the RAG calibration experiment.")
    parser.add_argument(
        "--question-sets",
        default="2025-10-26",
        help="Comma-separated ForecastBench question-set dates (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Cap on the number of questions to process.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=60,
        help="Tavily start_date offset before each question's freeze_datetime.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path; defaults to data/results/run_<timestamp>.csv.",
    )
    args = parser.parse_args()

    dates = tuple(d.strip() for d in args.question_sets.split(",") if d.strip())
    cfg = Config(question_set_dates=dates, lookback_days=args.lookback_days)
    out_path = (
        Path(args.out)
        if args.out
        else cfg.results_dir
        / f"run_{datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
    )
    asyncio.run(run(cfg, args.max_questions, out_path))


if __name__ == "__main__":
    main()
