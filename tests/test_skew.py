"""Batch and API must agree exactly. Not closely — exactly.

Train/serve skew is the silent killer in the plan's list, and it earns the name:
when it happens, nothing errors. The batch job and the API each build their input
slightly differently, predictions diverge on a few percent of rows, accuracy
drops, and every metric in the warehouse still looks plausible because both halves
are individually correct.

This is TokenWatch's reconciliation discipline applied to serving: two
independent routes to the same number must agree, or neither is trusted. The
assertion is bit-for-bit on the label *and* the confidence, because the Stage 4
cascade gates on confidence and a drift there would change escalation behaviour
without changing a single label.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kanal.features.text import Example, to_text
from kanal.models.tfidf import TfidfLinearSVC
from kanal.registry.artifact import save
from kanal.registry.store import Registry
from kanal.serving.api import app, get_loader
from kanal.serving.loader import ChampionLoader

TRAIN = [
    ("Presiden resmikan bendungan baru di Jawa Barat", "politik"),
    ("DPR sahkan undang-undang baru soal pemilu", "politik"),
    ("Menteri umumkan kebijakan subsidi energi terbaru", "politik"),
    ("Partai gelar kongres nasional di ibu kota", "politik"),
    ("Harga emas menguat di tengah ketidakpastian pasar", "ekonomi"),
    ("Bank sentral pertahankan suku bunga acuan", "ekonomi"),
    ("Rupiah melemah terhadap dolar Amerika Serikat", "ekonomi"),
    ("Inflasi tahunan tercatat naik tipis bulan lalu", "ekonomi"),
    ("Timnas futsal menang atas Thailand di partai final", "olahraga"),
    ("Persib segel tiket lolos ke babak semifinal", "olahraga"),
    ("Pemain muda cetak gol kemenangan di menit akhir", "olahraga"),
    ("Kompetisi liga utama dimulai akhir pekan ini", "olahraga"),
]

# Deliberately awkward: publisher boilerplate, unicode variants, extra
# whitespace, an empty summary. Skew hides in exactly these.
PROBE = [
    ("Harga emas kembali menguat hari ini", "Analis memperkirakan tren berlanjut."),
    ("Timnas menang telak di kandang lawan", "Liputan6.com, Jakarta - Pertandingan berakhir 3-0."),
    (
        "DPR bahas anggaran negara tahun depan",
        "REPUBLIKA.CO.ID, JAKARTA -- Rapat digelar tertutup.",
    ),
    ("  Rupiah   melemah \n tajam  ", ""),
    # Explicit escapes: the non-breaking spaces and curly quotes here are the
    # point of the row, and a literal NBSP is invisible in a diff.
    (
        "Menteri\u00a0“menolak”\u00a0usulan kenaikan",
        "Jakarta (ANTARA) - Pernyataan itu disampaikan.",
    ),
    ("Kompetisi dimulai akhir pekan", "Sebanyak 18 tim ikut serta."),
]


@pytest.fixture
def served(tmp_path: Path) -> tuple[TestClient, TfidfLinearSVC]:
    """A registry with one promoted model, and a client wired to it."""
    model = TfidfLinearSVC(min_df=1)
    model.fit([t for t, _ in TRAIN], [k for _, k in TRAIN])

    artifact_dir = tmp_path / "built"
    meta = save(model, artifact_dir, split_hash="split-under-test")

    registry = Registry(tmp_path / "registry")
    registry.register(artifact_dir, meta.id)
    registry.promote(meta.id)

    loader = ChampionLoader(tmp_path / "registry", ttl_seconds=0.0)
    app.dependency_overrides[get_loader] = lambda: loader
    yield TestClient(app), model
    app.dependency_overrides.clear()


class TestBatchAndApiAgree:
    def test_labels_are_identical(self, served: tuple[TestClient, TfidfLinearSVC]) -> None:
        client, model = served

        batch = model.predict([to_text(Example(title=t, summary=s)) for t, s in PROBE])
        response = client.post(
            "/predict",
            json={"articles": [{"title": t, "summary": s} for t, s in PROBE]},
        )
        assert response.status_code == 200
        served_labels = [p["kanal"] for p in response.json()["predictions"]]

        assert [p.label for p in batch] == served_labels

    def test_confidences_are_identical_to_the_reported_precision(
        self, served: tuple[TestClient, TfidfLinearSVC]
    ) -> None:
        """Not just the label.

        The cascade escalates below a confidence threshold, so a drift in the
        confidence changes escalation behaviour without changing any label — a
        skew that no label-only test could see.
        """
        client, model = served

        batch = model.predict([to_text(Example(title=t, summary=s)) for t, s in PROBE])
        response = client.post(
            "/predict",
            json={"articles": [{"title": t, "summary": s} for t, s in PROBE]},
        )
        served_conf = [p["confidence"] for p in response.json()["predictions"]]

        # The API rounds to 4dp for the wire; compare at that precision.
        assert [round(p.confidence, 4) for p in batch] == served_conf

    def test_full_probability_vectors_agree(
        self, served: tuple[TestClient, TfidfLinearSVC]
    ) -> None:
        client, model = served
        batch = model.predict([to_text(Example(title=t, summary=s)) for t, s in PROBE])
        served_probs = client.post(
            "/predict",
            json={"articles": [{"title": t, "summary": s} for t, s in PROBE]},
        ).json()["predictions"]

        for expected, actual in zip(batch, served_probs, strict=True):
            assert {k: round(v, 4) for k, v in expected.probabilities.items()} == actual[
                "probabilities"
            ]

    def test_a_larger_sample_agrees_row_for_row(
        self, served: tuple[TestClient, TfidfLinearSVC]
    ) -> None:
        # 200 rows through both paths, which is the plan's number. Generated by
        # varying the probe set so the sample is not six rows repeated.
        client, model = served
        rows = [(f"{title} nomor {i}", summary) for i in range(34) for title, summary in PROBE][
            :200
        ]

        batch = model.predict([to_text(Example(title=t, summary=s)) for t, s in rows])
        served_out: list[dict[str, object]] = []
        for start in range(0, len(rows), 64):
            chunk = rows[start : start + 64]
            served_out.extend(
                client.post(
                    "/predict",
                    json={"articles": [{"title": t, "summary": s} for t, s in chunk]},
                ).json()["predictions"]
            )

        assert len(served_out) == 200
        assert [p.label for p in batch] == [p["kanal"] for p in served_out]
        assert [round(p.confidence, 4) for p in batch] == [p["confidence"] for p in served_out]


class TestTheApiSurface:
    def test_the_response_names_the_model_that_answered(
        self, served: tuple[TestClient, TfidfLinearSVC]
    ) -> None:
        # So a prediction can be traced back to the promotion decision that put
        # that model in place.
        client, _ = served
        body = client.post("/predict", json={"articles": [{"title": "Harga emas naik"}]}).json()

        assert body["model_id"]
        assert body["model_name"] == "tfidf-linearsvc"
        assert body["latency_ms"] > 0
        assert body["usd_per_1000"] > 0, "self-hosted is not free"

    def test_health_reports_what_is_serving(
        self, served: tuple[TestClient, TfidfLinearSVC]
    ) -> None:
        client, _ = served
        body = client.get("/health").json()

        assert body["status"] == "ok"
        assert body["split_hash"] == "split-under-test"
        assert set(body["classes"]) == {"politik", "ekonomi", "olahraga"}
        assert body["failed_reloads"] == 0

    def test_an_empty_batch_is_rejected(self, served: tuple[TestClient, TfidfLinearSVC]) -> None:
        assert client_post(served, {"articles": []}) == 422

    def test_an_oversized_batch_is_rejected(
        self, served: tuple[TestClient, TfidfLinearSVC]
    ) -> None:
        articles = [{"title": f"judul {i}"} for i in range(200)]
        assert client_post(served, {"articles": articles}) == 422

    def test_a_blank_title_is_rejected(self, served: tuple[TestClient, TfidfLinearSVC]) -> None:
        assert client_post(served, {"articles": [{"title": ""}]}) == 422

    def test_provenance_fields_are_not_accepted_as_input(
        self, served: tuple[TestClient, TfidfLinearSVC]
    ) -> None:
        """The leakage defence, at the API boundary.

        A caller cannot pass a URL or a source even by accident — and if they
        try, the field is ignored rather than reaching a feature.
        """
        client, _ = served
        body = client.post(
            "/predict",
            json={
                "articles": [
                    {
                        "title": "Harga emas naik",
                        "canonical_url": "https://cnnindonesia.com/ekonomi/xyz",
                        "source": "cnn",
                    }
                ]
            },
        )
        assert body.status_code == 200
        # It classified from the title alone; the URL naming 'ekonomi' had no
        # route into the feature path.
        assert "kanal" in body.json()["predictions"][0]


def client_post(served: tuple[TestClient, TfidfLinearSVC], payload: dict[str, object]) -> int:
    client, _ = served
    return client.post("/predict", json=payload).status_code


class TestNoChampion:
    def test_serving_without_a_promotion_is_503_not_500(self, tmp_path: Path) -> None:
        """The distinction decides whether someone hunts a bug or runs a job."""
        loader = ChampionLoader(tmp_path / "empty", ttl_seconds=0.0)
        app.dependency_overrides[get_loader] = lambda: loader
        try:
            client = TestClient(app)
            response = client.post("/predict", json={"articles": [{"title": "x"}]})
            assert response.status_code == 503
            assert "nothing to serve" in response.json()["detail"]
            assert client.get("/health").json()["status"] == "no_champion"
        finally:
            app.dependency_overrides.clear()
