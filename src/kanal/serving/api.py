"""The prediction API.

Every response says which model answered, how confident it was, what the call
cost, and how long it took. That is not decoration — it is the same discipline
TokenWatch used, and it is what makes the Stage 4 cascade legible when a cheap
model starts escalating more often.

The serving path builds its input through `kanal.features.text.to_text` and
nothing else. That is the same function the training path calls, and
`tests/test_skew.py` pushes rows through both and asserts the predictions are
identical rather than close.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from kanal.features.text import Example, to_text
from kanal.registry.artifact import Artifact
from kanal.serving.loader import ChampionLoader, NoChampion

MAX_BATCH = 64


def registry_root() -> Path:
    return Path(os.environ.get("KANAL_REGISTRY", "data/registry"))


_loader: ChampionLoader | None = None


def get_loader() -> ChampionLoader:
    global _loader
    if _loader is None:
        _loader = ChampionLoader(registry_root())
    return _loader


class ArticleIn(BaseModel):
    title: str = Field(min_length=1, max_length=500, examples=["Harga emas menguat"])
    summary: str = Field(default="", max_length=2000)


class PredictIn(BaseModel):
    articles: list[ArticleIn] = Field(min_length=1, max_length=MAX_BATCH)


class PredictionOut(BaseModel):
    kanal: str
    confidence: float
    probabilities: dict[str, float]


class PredictOut(BaseModel):
    predictions: list[PredictionOut]
    # Which model answered, so a response can be traced to a promotion decision.
    model_id: str
    model_name: str
    latency_ms: float
    usd_per_1000: float


class HealthOut(BaseModel):
    status: str
    model_id: str | None
    model_name: str | None
    classes: list[str]
    split_hash: str | None
    reloads: int
    failed_reloads: int
    last_error: str | None


app = FastAPI(
    title="KANAL",
    version="0.3.0",
    description=(
        "Classifies Indonesian news articles into one of eight sections from the "
        "headline and RSS summary alone — never the article body.\n\n"
        "Every response names the model that answered and what the call cost. "
        "The champion alias is re-read on a timer, so a rollback takes effect "
        "within 60 seconds with no redeploy."
    ),
)


Loader = Annotated[ChampionLoader, Depends(get_loader)]


def _current(loader: ChampionLoader) -> Artifact:
    try:
        return loader.get()
    except NoChampion as err:
        # 503 rather than 500: the service is correctly configured and simply has
        # nothing promoted yet. That distinction decides whether someone goes
        # looking for a bug or runs the promotion job.
        raise HTTPException(status_code=503, detail=str(err)) from err


@app.get("/health", response_model=HealthOut)
def health(loader: Loader) -> HealthOut:
    """What is serving right now, and whether the last reload worked.

    An operator should be able to confirm a promotion took effect here rather
    than inferring it from prediction quality.
    """
    state = loader.state
    try:
        artifact = loader.get()
    except NoChampion:
        return HealthOut(
            status="no_champion",
            model_id=None,
            model_name=None,
            classes=[],
            split_hash=None,
            reloads=state.reloads,
            failed_reloads=state.failed_reloads,
            last_error=state.last_error,
        )

    return HealthOut(
        # "degraded" when a reload failed: the model being served is still
        # working, but it is not the one the registry says should be.
        status="degraded" if state.last_error else "ok",
        model_id=artifact.meta.id,
        model_name=artifact.meta.name,
        classes=artifact.meta.classes,
        split_hash=artifact.meta.split_hash,
        reloads=state.reloads,
        failed_reloads=state.failed_reloads,
        last_error=state.last_error,
    )


@app.post("/predict", response_model=PredictOut)
def predict(body: PredictIn, loader: Loader) -> PredictOut:
    """Classify one or more articles.

    Input is the headline and optional summary. Nothing else is accepted — the
    URL and the source are not parameters, because a model that sees them scores
    near-perfectly and has learnt nothing.
    """
    artifact = _current(loader)

    # The single feature path, shared with training.
    texts = [to_text(Example(title=a.title, summary=a.summary)) for a in body.articles]

    start = time.perf_counter()
    predictions = artifact.model.predict(texts)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    try:
        cost = artifact.model.unit_cost_usd(
            1000, measured_seconds_per_prediction=(elapsed_ms / len(texts)) / 1000.0
        )
    except TypeError:
        cost = artifact.model.unit_cost_usd(1000)

    return PredictOut(
        predictions=[
            PredictionOut(
                kanal=p.label,
                confidence=round(p.confidence, 4),
                probabilities={k: round(v, 4) for k, v in p.probabilities.items()},
            )
            for p in predictions
        ],
        model_id=artifact.meta.id,
        model_name=artifact.meta.name,
        latency_ms=round(elapsed_ms, 3),
        usd_per_1000=cost,
    )


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "KANAL",
        "docs": "/docs",
        "health": "/health",
        "dataset": "https://huggingface.co/datasets/aeroo11/kanal-berita",
        "source": "https://github.com/Aeroo11/kanal-berita-dashboard",
    }
