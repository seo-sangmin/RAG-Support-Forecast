from __future__ import annotations

import asyncio
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Config
from .data import ResolvedQuestion, load_resolved_questions
from .forecasting import ForecastClient
from .metrics import brier, z_tentori_crupi
from .retrieval import TavilyRetriever


@dataclass
class Row:
    id: str
    source: str
    question: str
    freeze_datetime: str
    resolution_date: str
    outcome: float
    p_h: float
    p_he: float
    n_evidence: int
    brier_h: float
    brier_he: float
    brier_delta: float
    z: float
    abs_z: float


async def _process(
    q: ResolvedQuestion,
    forecaster: ForecastClient,
    retriever: TavilyRetriever,
    sem: asyncio.Semaphore,
) -> Row | None:
    async with sem:
        try:
            prior = await forecaster.estimate_p_h(q)
            evidence = await retriever.retrieve(q)
            posterior = await forecaster.estimate_p_h_given_e(q, evidence)
        except Exception as e:  # noqa: BLE001
            print(f"  ! skipped {q.id}: {e}")
            return None

    p_h = prior["probability"]
    p_he = posterior["probability"]
    bh = brier(p_h, q.outcome)
    bhe = brier(p_he, q.outcome)
    z = z_tentori_crupi(p_h, p_he)
    return Row(
        id=q.id,
        source=q.source,
        question=q.question,
        freeze_datetime=q.freeze_datetime.isoformat(),
        resolution_date=q.resolution_date,
        outcome=q.outcome,
        p_h=p_h,
        p_he=p_he,
        n_evidence=len(evidence),
        brier_h=bh,
        brier_he=bhe,
        brier_delta=bh - bhe,
        z=z,
        abs_z=abs(z),
    )


async def run(cfg: Config, max_questions: int | None, out_csv: Path) -> int:
    questions: list[ResolvedQuestion] = []
    for date in cfg.question_set_dates:
        questions.extend(load_resolved_questions(date, cfg))
    if max_questions is not None:
        questions = questions[:max_questions]

    print(f"Loaded {len(questions)} resolved binary questions")
    forecaster = ForecastClient(cfg)
    retriever = TavilyRetriever(cfg)
    sem = asyncio.Semaphore(cfg.concurrency)

    tasks = [_process(q, forecaster, retriever, sem) for q in questions]
    rows: list[Row] = []
    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        row = await coro
        if row is not None:
            rows.append(row)
        if i % 5 == 0 or i == len(tasks):
            print(f"  progress: {i}/{len(tasks)}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(Row.__annotations__.keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
    print(f"Wrote {len(rows)} rows to {out_csv}")
    return len(rows)
