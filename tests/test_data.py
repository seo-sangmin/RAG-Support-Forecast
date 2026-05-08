from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_forecast.config import Config
from rag_forecast.data import load_resolved_questions


def _write_fixtures(raw_dir: Path, date: str) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    questions = {
        "forecast_due_date": date,
        "question_set": f"{date}-llm.json",
        "questions": [
            {
                "id": "q1",
                "source": "manifold",
                "question": "Will X happen?",
                "resolution_criteria": "Resolves YES if X happens.",
                "background": "Some context.",
                "freeze_datetime": "2025-10-16T00:00:00+00:00",
                "freeze_datetime_value": 0.42,
            },
            {
                "id": "q2",
                "source": "fred",
                "question": "Will Y exceed threshold?",
                "resolution_criteria": "Resolves YES if Y > threshold.",
                "background": "",
                "freeze_datetime": "2025-10-16T00:00:00+00:00",
                "freeze_datetime_value": None,
            },
            {
                "id": "q3",
                "source": "manifold",
                "question": "Will Z drop?",
                "resolution_criteria": "",
                "background": "",
                "freeze_datetime": "2025-10-16T00:00:00+00:00",
                "freeze_datetime_value": 0.1,
            },
        ],
    }
    resolutions = {
        "forecast_due_date": date,
        "question_set": f"{date}-llm.json",
        "resolutions": [
            {
                "id": "q1",
                "source": "manifold",
                "direction": None,
                "resolution_date": "2025-12-01",
                "resolved_to": 1.0,
                "resolved": True,
            },
            {
                "id": "q2",
                "source": "fred",
                "direction": None,
                "resolution_date": "2025-12-15",
                "resolved_to": 0.0,
                "resolved": True,
            },
            # q3 is unresolved -> must be filtered out
            {
                "id": "q3",
                "source": "manifold",
                "direction": None,
                "resolution_date": None,
                "resolved_to": None,
                "resolved": False,
            },
        ],
    }
    (raw_dir / f"{date}-llm.json").write_text(json.dumps(questions))
    (raw_dir / f"{date}_resolution_set.json").write_text(json.dumps(resolutions))


def test_load_resolved_questions_filters_and_joins(tmp_path: Path) -> None:
    cfg = Config(raw_dir=tmp_path / "raw", cache_dir=tmp_path / "cache",
                 results_dir=tmp_path / "results")
    _write_fixtures(cfg.raw_dir, "2025-10-26")

    out = load_resolved_questions("2025-10-26", cfg)
    assert len(out) == 2
    by_id = {q.id: q for q in out}
    assert by_id["q1"].outcome == 1.0
    assert by_id["q1"].source == "manifold"
    assert by_id["q1"].freeze_value == 0.42
    assert by_id["q2"].outcome == 0.0
    assert "q3" not in by_id
