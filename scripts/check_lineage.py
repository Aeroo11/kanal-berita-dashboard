"""Assert the asset graph is one connected DAG, and print it.

This exists because of a failure that passes every other check. dbt declares its
source as `kanal.raw_articles`, which dagster-dbt turns into the asset key
`kanal/raw_articles` — while the Python asset that writes that table is keyed
`raw_articles`. Both nodes then exist, nothing connects them, and the lineage
view shows two clusters that each look perfectly healthy.

Every asset materialises. Every check passes. And the single reason an
orchestrator was added here — being able to trace a number in a mart back to the
feed it came from — stops working, silently.

So: walk from the ingestion root and require that everything is reachable.

    python scripts/check_lineage.py
"""

from __future__ import annotations

import sys
from collections import defaultdict

from kanal.orchestration.definitions import defs

ROOT = "rss_landing_zone"

# Roots that legitimately have no upstream: seeds are checked into the repo
# rather than produced by the pipeline.
ALLOWED_EXTRA_ROOTS = {"taxonomy_map"}


def main() -> int:
    repo = defs.get_repository_def()

    upstream: dict[str, set[str]] = defaultdict(set)
    groups: dict[str, str] = {}
    keys: set[str] = set()

    for assets_def in repo.assets_defs_by_key.values():
        for spec in assets_def.specs:
            key = spec.key.to_user_string()
            keys.add(key)
            groups[key] = spec.group_name or "-"
            for dep in spec.deps:
                upstream[key].add(dep.asset_key.to_user_string())

    # Print the graph regardless of outcome; the shape is the artifact.
    print(f"{len(keys)} assets\n")
    for key in sorted(keys, key=lambda k: (groups[k], k)):
        deps = sorted(upstream[key])
        suffix = "  <- " + ", ".join(deps) if deps else "  (root)"
        print(f"  [{groups[key]:<12}] {key}{suffix}")

    problems: list[str] = []

    if ROOT not in keys:
        problems.append(f"the ingestion root {ROOT!r} is missing from the graph")

    # Any asset whose declared upstream is not itself an asset means a dangling
    # dependency — the exact symptom of the split-key bug.
    for key, deps in upstream.items():
        for dep in deps:
            if dep not in keys:
                problems.append(f"{key} depends on {dep!r}, which is not an asset in this graph")

    # Everything must be reachable from the ingestion root, seeds aside.
    downstream: dict[str, set[str]] = defaultdict(set)
    for key, deps in upstream.items():
        for dep in deps:
            downstream[dep].add(key)

    reachable: set[str] = set()
    stack = [ROOT, *ALLOWED_EXTRA_ROOTS]
    while stack:
        node = stack.pop()
        if node in reachable:
            continue
        reachable.add(node)
        stack.extend(downstream.get(node, ()))

    orphans = sorted(keys - reachable)
    if orphans:
        problems.append("not reachable from the ingestion root: " + ", ".join(orphans))

    print()
    if problems:
        for p in problems:
            print(f"FAIL  {p}", file=sys.stderr)
        return 1

    print(f"OK  one connected graph, {len(keys)} assets reachable from {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
