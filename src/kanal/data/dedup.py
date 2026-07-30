"""Near-duplicate clustering by MinHash / LSH.

The warehouse already clusters on an exact normalised-title hash, which catches
republication that preserves the headline — the common case for wire copy. It
misses the harder one: the same event written up independently, or a headline
lightly rewritten by a subeditor.

    ANTARA      "Pemerintah naikkan cukai rokok 10 persen tahun depan"
    Liputan6    "Cukai Rokok Naik 10 Persen pada 2027, Ini Alasannya"

Different strings, same story. Split by row and one lands in train while the
other lands in test — and the evaluation then measures memorisation.

Written by hand rather than with `datasketch`, for the same reason ingestion is
plain Python (ADR-002): the shingling and banding *are* the thing being
demonstrated, and a library import cannot be explained in an interview. It is
also about eighty lines.

## How it works

1. **Shingle.** Each title becomes a set of character 4-grams over its
   normalised form. Character n-grams rather than word n-grams because
   Indonesian is agglutinative — `menaikkan`, `kenaikan` and `naik` share a stem
   that word tokens split apart, and character grams keep the overlap.

2. **MinHash.** Hash every shingle with `k` independent hash functions and keep
   the minimum per function. The probability that two documents agree on any one
   position equals their Jaccard similarity — that identity is what makes the
   signature a usable stand-in for the full set.

3. **Band.** Split the signature into `b` bands of `r` rows. Two documents become
   candidates if any band matches exactly. The probability of that is
   `1 - (1 - s^r)^b` — an S-curve whose knee sits near `(1/b)^(1/r)`, which is how
   the threshold is chosen rather than guessed.

4. **Union-find.** Candidate pairs above the similarity threshold are merged into
   transitive clusters, so A~B and B~C puts all three together.

## What this cannot do, measured

Lexical near-duplicate detection has a real ceiling on this corpus, and it is
worth stating with numbers rather than discovering later.

Indonesian news headlines are heavily templated — "Link Live Streaming X vs Y di
Piala Presiden 2026" is a form, not a story — so character n-grams over the
template score high even for unrelated matches. Measured on hand-labelled pairs
from the real corpus:

    true duplicates   Jaccard 0.263 – 0.688
    template collisions       0.379 – 0.575

**Those distributions overlap, so no threshold separates them.** Two other ideas
were tried and measured, and both failed:

- *Distinctive-token overlap* (capitalised words and numbers, on the theory that
  different matches name different teams): true 0.111–0.750, false 0.364–0.600.
  Also overlapping.
- *Including the summary*: worse, not better. Publishers write their own summaries
  for the same event, so lexical overlap is low for genuine duplicates too —
  "Korban Gempa Kumamoto Bertambah, 17 Orang Tewas" and "Korban Jiwa Gempa
  Kumamoto Bertambah jadi 13 Orang" are the same developing story with a summary
  similarity of 0.09.

So the residual error is accepted rather than engineered away, because **the two
error directions are not symmetric.** This clustering exists to keep the same
story out of both train and test:

- *over-merging* two different stories costs a little test-set diversity;
- *under-merging* the same story lets memorisation be scored as generalisation,
  which invalidates the evaluation entirely.

The threshold is therefore set to favour recall. On the corpus at the time of
writing that produced 16 multi-row clusters covering 42 of 1,295 rows, of which
hand inspection judged 2–3 to be genuine over-merges — roughly 3% of the corpus
touched, erring in the harmless direction.

There is a second ceiling, and it is the banding rather than the threshold. With
32 bands of 4 rows the S-curve knee sits near `(1/32)^(1/4) ≈ 0.42`, so a pair at
0.26 similarity has roughly a 14% chance of sharing any band — it is never
*offered* for verification, and lowering the threshold cannot help. Re-banding to
reach it would flood a candidate set already running at a 98% false rate on the
real corpus, for pairs as likely to be template collisions as duplicates.

A semantic fix (sentence embeddings rather than shingles) would separate these
properly, and an embedding model arrives in a later stage for drift detection.
Until then this is a documented limitation, not an unknown one.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

# Character 4-grams. Short enough to survive a rewritten headline, long enough
# that common Indonesian function words ("yang", "dan") do not dominate.
SHINGLE_SIZE = 4

# 128 permutations in 32 bands of 4 rows. The knee of the S-curve sits at
# (1/32)^(1/4) ≈ 0.42, so pairs above ~0.5 Jaccard are reliably caught and pairs
# below ~0.3 reliably are not.
NUM_PERM = 128
NUM_BANDS = 32
ROWS_PER_BAND = NUM_PERM // NUM_BANDS

# Verified exactly on the candidate pairs LSH surfaces. LSH is a filter, not a
# decision: banding produces false positives by design and the real similarity is
# cheap to compute once the candidate set is small.
SIMILARITY_THRESHOLD = 0.5

_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")

_MERSENNE = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return _WHITESPACE.sub(" ", _NON_WORD.sub(" ", text.lower())).strip()


def shingles(text: str, size: int = SHINGLE_SIZE) -> set[str]:
    """Character n-grams of the normalised text.

    A title shorter than the shingle size yields itself, so it can still be
    compared rather than silently producing an empty signature that matches
    everything.
    """
    norm = normalise(text)
    if len(norm) < size:
        return {norm} if norm else set()
    return {norm[i : i + size] for i in range(len(norm) - size + 1)}


def _hash32(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=4).digest(), "big")


@dataclass(frozen=True, slots=True)
class Permutations:
    """Coefficients for `k` independent hash functions.

    Universal hashing: `h_i(x) = (a_i * x + b_i) mod p`, with p a Mersenne prime
    larger than the hash space. Seeded, so signatures are reproducible across
    runs and machines — a clustering that shifted between runs would make the
    frozen split manifest worthless.
    """

    a: tuple[int, ...]
    b: tuple[int, ...]

    @classmethod
    def seeded(cls, seed: int = 42, k: int = NUM_PERM) -> Permutations:
        # Derived from the seed by hashing rather than a PRNG, so the values do
        # not depend on which Python version's random module is in use.
        a: list[int] = []
        b: list[int] = []
        for i in range(k):
            a.append((_hash32(f"a{seed}:{i}") | 1) % _MERSENNE)  # odd, so never degenerate
            b.append(_hash32(f"b{seed}:{i}") % _MERSENNE)
        return cls(tuple(a), tuple(b))


def signature(text: str, perms: Permutations) -> tuple[int, ...]:
    """MinHash signature: the minimum of each hash function over the shingles."""
    grams = shingles(text)
    if not grams:
        return tuple([_MAX_HASH] * len(perms.a))

    hashed = [_hash32(g) for g in grams]
    return tuple(
        min((a * h + b) % _MERSENNE for h in hashed) for a, b in zip(perms.a, perms.b, strict=True)
    )


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


class _UnionFind:
    """Transitive merging, so A~B and B~C puts all three in one cluster."""

    def __init__(self, keys: Iterable[str]) -> None:
        self._parent = {k: k for k in keys}

    def find(self, key: str) -> str:
        root = key
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression: keeps repeated lookups near-constant.
        while self._parent[key] != root:
            self._parent[key], key = root, self._parent[key]
        return root

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            # Lexicographic root, so cluster ids are deterministic rather than
            # depending on the order rows arrived in.
            lo, hi = sorted((a, b))
            self._parent[hi] = lo

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for key in self._parent:
            out.setdefault(self.find(key), []).append(key)
        return out


@dataclass
class ClusterReport:
    """Cluster assignment plus what the clustering actually found."""

    cluster_of: dict[str, str]
    candidate_pairs: int = 0
    confirmed_pairs: int = 0
    clusters: int = 0
    largest_cluster: int = 0
    rows_in_a_cluster: int = 0
    sizes: dict[int, int] = field(default_factory=dict)

    @property
    def false_candidate_rate(self) -> float:
        """Share of LSH candidates that did not survive exact verification.

        Worth reporting: a rate near zero suggests the threshold is too loose to
        be finding anything, and a rate near one suggests the bands are too wide
        and most of the work is wasted.
        """
        if not self.candidate_pairs:
            return 0.0
        return 1.0 - (self.confirmed_pairs / self.candidate_pairs)

    def summary(self) -> str:
        return (
            f"{self.clusters} clusters over {len(self.cluster_of)} rows "
            f"({self.rows_in_a_cluster} in a group of >1, largest {self.largest_cluster}); "
            f"{self.confirmed_pairs}/{self.candidate_pairs} candidate pairs confirmed"
        )


def cluster(
    rows: Sequence[tuple[str, str]],
    *,
    threshold: float = SIMILARITY_THRESHOLD,
    seed: int = 42,
) -> ClusterReport:
    """Cluster `(key, title)` pairs by near-duplicate title.

    Returns a mapping from key to cluster id, where the id is the lexicographically
    smallest key in the cluster — deterministic, and stable if rows are reordered.
    """
    perms = Permutations.seeded(seed)
    keys = [key for key, _ in rows]

    grams = {key: shingles(title) for key, title in rows}
    sigs = {key: signature(title, perms) for key, title in rows}

    # Band the signatures: identical band → candidate pair.
    buckets: dict[tuple[int, tuple[int, ...]], list[str]] = {}
    for key in keys:
        sig = sigs[key]
        for band in range(NUM_BANDS):
            chunk = sig[band * ROWS_PER_BAND : (band + 1) * ROWS_PER_BAND]
            buckets.setdefault((band, chunk), []).append(key)

    candidates: set[tuple[str, str]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        # A band shared by very many rows is a degenerate signature — near-empty
        # titles, or a shingle set so small every hash collapses. Pairing them all
        # would be quadratic and meaningless.
        if len(members) > 50:
            continue
        for i, left in enumerate(members):
            for right in members[i + 1 :]:
                candidates.add((left, right) if left < right else (right, left))

    uf = _UnionFind(keys)
    confirmed = 0
    for left, right in candidates:
        if jaccard(grams[left], grams[right]) >= threshold:
            uf.union(left, right)
            confirmed += 1

    groups = uf.groups()
    cluster_of = {key: root for root, members in groups.items() for key in members}

    sizes: dict[int, int] = {}
    for members in groups.values():
        sizes[len(members)] = sizes.get(len(members), 0) + 1

    return ClusterReport(
        cluster_of=cluster_of,
        candidate_pairs=len(candidates),
        confirmed_pairs=confirmed,
        clusters=len(groups),
        largest_cluster=max((len(m) for m in groups.values()), default=0),
        rows_in_a_cluster=sum(len(m) for m in groups.values() if len(m) > 1),
        sizes=sizes,
    )
