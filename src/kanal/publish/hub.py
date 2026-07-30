"""Upload the export to a Hugging Face dataset repository.

The dataset is the deliverable of Stage 1: a growing, versioned, publicly
inspectable corpus of Indonesian news headlines that someone else can actually
use. Hugging Face rather than a bucket because every commit is a reproducible
snapshot — data versioning for free — and because the dataset viewer means a
visitor can look at the rows without downloading anything.

The token is read from the environment only. Never a flag, so it cannot end up in
a shell history or a CI log line.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_REPO = "aeroo11/kanal-berita"


class MissingTokenError(RuntimeError):
    """Raised when no write token is available."""

    def __init__(self) -> None:
        super().__init__(
            "No Hugging Face token found. Set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) "
            "to a token with write access. Create one at "
            "https://huggingface.co/settings/tokens — it needs the 'write' scope."
        )


@dataclass
class UploadReport:
    repo_id: str
    url: str
    files: list[str]
    commit_url: str | None

    def summary(self) -> str:
        return f"pushed {len(self.files)} file(s) to {self.repo_id} → {self.url}"


def _token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise MissingTokenError
    return token


def upload(
    export_dir: Path | None = None,
    repo_id: str | None = None,
    *,
    private: bool = False,
    dry_run: bool = False,
) -> UploadReport:
    """Push `articles.parquet`, the dataset card and the stats sidecar."""
    from huggingface_hub import HfApi

    source = export_dir or (Path("data") / "export")
    repo = repo_id or os.environ.get("KANAL_HF_REPO") or DEFAULT_REPO

    expected = ["articles.parquet", "README.md", "stats.json"]
    missing = [name for name in expected if not (source / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"{', '.join(missing)} not found in {source}. Run `kanal export` first."
        )

    url = f"https://huggingface.co/datasets/{repo}"

    if dry_run:
        log.info("dry run: would push %s to %s", ", ".join(expected), repo)
        return UploadReport(repo_id=repo, url=url, files=expected, commit_url=None)

    api = HfApi(token=_token())
    api.create_repo(repo_id=repo, repo_type="dataset", private=private, exist_ok=True)

    # One commit for the whole export rather than three. A card that lands in a
    # separate commit from the Parquet it describes leaves a window where the
    # published numbers do not match the published file.
    info = api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=str(source),
        allow_patterns=expected,
        commit_message="Update dataset export",
    )

    return UploadReport(
        repo_id=repo,
        url=url,
        files=expected,
        commit_url=getattr(info, "commit_url", None),
    )
