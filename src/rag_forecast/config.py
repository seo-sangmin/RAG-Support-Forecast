from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

FORECASTBENCH_QUESTION_URL = (
    "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/"
    "main/datasets/question_sets/{date}-llm.json"
)
FORECASTBENCH_RESOLUTION_URL = (
    "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/"
    "main/datasets/resolution_sets/{date}_resolution_set.json"
)


@dataclass(frozen=True)
class Config:
    model: str = "claude-haiku-4-5-20251001"
    temperature: float = 0.0
    max_tokens: int = 1024

    question_set_dates: tuple[str, ...] = ("2025-10-26",)
    lookback_days: int = 60
    tavily_max_results: int = 8
    tavily_search_depth: str = "advanced"
    tavily_snippet_chars: int = 2000

    concurrency: int = 8
    llm_max_retries: int = 2

    raw_dir: Path = field(default_factory=lambda: REPO_ROOT / "data" / "raw")
    cache_dir: Path = field(default_factory=lambda: REPO_ROOT / "data" / "cache")
    results_dir: Path = field(default_factory=lambda: REPO_ROOT / "data" / "results")
