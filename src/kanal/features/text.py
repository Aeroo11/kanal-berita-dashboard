"""The single path from a row to model input.

Two separate disasters are prevented here, and both are silent.

## Train/serve skew

If training builds its text one way and the API builds it another — a stripped
prefix here, a lowercase there — the model sees a different distribution in
production than it was fitted on. Accuracy drops, every metric in the warehouse
still looks fine, and nothing in the system is wrong enough to raise an alarm.

The defence is structural rather than disciplinary: there is one function, both
paths import it, and a CI test pushes rows through the batch path and the API
path and asserts the predictions are identical. Not "close" — identical.

## Label leakage

The label *is* feed provenance, and it leaks into the URL: CNN's article URLs
contain their section 100% of the time, Liputan6's 98.6%, Republika's 35.6%,
ANTARA's 4.1%. A model that sees any of that scores near-perfectly and has
learnt nothing.

So the input to `to_text` is a deliberately narrow struct. It carries the title
and the summary, and it does not carry the URL, the source, the feed, the
channel, the timestamps, or the cluster id — not as ignored fields, but as fields
that are not there to be read. Leaking one becomes a type error rather than a
silently better score.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# What a feature may be built from. Anything else is a leak.
ALLOWED_FIELDS = ("title", "summary")

# Recorded so the leakage test can assert on it, and so the list is reviewable
# rather than implicit in a struct definition. Every one of these correlates
# with the label by construction rather than by meaning.
FORBIDDEN_FIELDS = (
    "article_key",
    "canonical_url",
    "raw_link",
    "source",
    "channel",
    "feed_id",
    "kanal",
    "cluster_id",
    "published_at",
    "fetched_at",
    "url_leaks_channel",
    "url_leaks_canonical",
    "url_leaks_label",
    "is_evergreen",
)

_WHITESPACE = re.compile(r"\s+")

# Publisher boilerplate that survives into the summary. Left in, a model can
# learn "starts with Liputan6.com" as a proxy for the outlet, and the outlet
# correlates with the label — leakage arriving by a second route.
_BOILERPLATE = re.compile(
    r"^(?:"
    r"liputan6\.com\s*,\s*[^-–—]{0,40}[-–—]+\s*"
    r"|republika\.co\.id\s*,\s*[^-–—]{0,40}[-–—]+\s*"
    r"|jakarta\s*,?\s*\(antara\w*\)\s*[-–—]+\s*"
    r"|antara\w*\s*[-–—]+\s*"
    r"|cnn\s+indonesia\s*[-–—]+\s*"
    r")",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Example:
    """Everything a model is permitted to see about an article.

    Deliberately minimal. The fields in `FORBIDDEN_FIELDS` are absent rather than
    present-and-ignored, so leaking one is a type error at the call site instead
    of a quietly better number three weeks later.
    """

    title: str
    summary: str = ""


def _strip_accents_of_control_chars(text: str) -> str:
    """Normalise unicode and drop control characters.

    NFKC folds the typographic variants publishers use inconsistently — curly
    quotes, non-breaking spaces, full-width punctuation — so the same headline
    from two outlets produces the same tokens.
    """
    normalised = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in normalised if unicodedata.category(ch)[0] != "C" or ch == "\n")


def to_text(example: Example) -> str:
    """Build the single string a model is fitted on and served from.

    Lowercasing is left to the vectoriser rather than done here, so that a
    candidate which benefits from casing — a transformer, later — can still see
    it. Everything genuinely destructive happens once, in this function.
    """
    title = _WHITESPACE.sub(" ", _strip_accents_of_control_chars(example.title)).strip()
    summary = _WHITESPACE.sub(" ", _strip_accents_of_control_chars(example.summary)).strip()
    summary = _BOILERPLATE.sub("", summary).strip()

    if not summary:
        return title
    if not title:
        return summary
    return f"{title}. {summary}"
