"""Command-line entry point.

kanal ingest [--source antara] [--dry-run]
kanal status
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from kanal.config import settings
from kanal.ingest.land import count_articles
from kanal.ingest.run import run_cycle
from kanal.ingest.sources import ALL_FEEDS, SOURCE_BY_NAME, feeds_for


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)-28s %(message)s",
        stream=sys.stderr,
    )


def _cmd_ingest(args: argparse.Namespace) -> int:
    feeds = feeds_for(args.source)
    print(f"polling {len(feeds)} feed(s)…", file=sys.stderr)

    report = run_cycle(feeds, sleep=not args.no_delay)
    print(report.summary())

    # A cycle that lost a whole publisher exits non-zero so a scheduled run
    # shows up red rather than quietly succeeding with a gap in the data.
    if not report.healthy:
        return 1
    return 0


def _cmd_load(args: argparse.Namespace) -> int:
    from kanal.warehouse.loader import default_db_path, load

    report = load(force=args.force)
    print(report.summary())
    print(f"warehouse: {default_db_path()}")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from kanal.publish.card import write_card
    from kanal.publish.export import export

    report = export(out_dir=args.out)
    print(report.summary())
    print(f"  articles  : {report.articles:,}")
    print(f"  sources   : {report.sources}")
    print(f"  classes   : {report.kanal_classes}")
    print(f"  published : {report.oldest} → {report.newest}")

    card = write_card(report.stats_path)
    print(f"  card      : {card}")
    return 0


def _cmd_publish(args: argparse.Namespace) -> int:
    from kanal.publish.hub import MissingTokenError, upload

    try:
        report = upload(
            export_dir=args.export_dir,
            repo_id=args.repo,
            private=args.private,
            dry_run=args.dry_run,
        )
    except MissingTokenError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4

    print(report.summary())
    if report.commit_url:
        print(f"  commit: {report.commit_url}")
    return 0


def _cmd_status(_: argparse.Namespace) -> int:
    total = count_articles()
    print(f"landing zone : {settings.raw_dir}")
    print(f"articles     : {total:,}")
    print(f"feeds        : {len(ALL_FEEDS)} across {len(SOURCE_BY_NAME)} sources")

    for name, source in sorted(SOURCE_BY_NAME.items()):
        leaks = []
        if source.section_in_url:
            leaks.append("url")
        if source.item_level_category:
            leaks.append("category")
        marker = f" [leaks: {'+'.join(leaks)}]" if leaks else " [clean]"
        print(f"  {name:<10} {len(source.feeds):>2} feeds{marker}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kanal", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="poll every feed once and land what is new")
    ingest.add_argument("--source", help="limit to one source (antara | cnn | liputan6)")
    ingest.add_argument(
        "--no-delay",
        action="store_true",
        help="skip the inter-request pause (tests only — do not use against live feeds)",
    )
    ingest.set_defaults(func=_cmd_ingest)

    load_cmd = sub.add_parser("load", help="load the landing zone into DuckDB")
    load_cmd.add_argument(
        "--force",
        action="store_true",
        help="re-read files already recorded as loaded (repairs a corrupted load log; "
        "the anti-join still prevents duplicate rows)",
    )
    load_cmd.set_defaults(func=_cmd_load)

    export_cmd = sub.add_parser(
        "export",
        help="write the modelled dataset to Parquet plus a generated dataset card",
    )
    export_cmd.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: data/export)",
    )
    export_cmd.set_defaults(func=_cmd_export)

    publish = sub.add_parser(
        "publish",
        help="upload the export to a Hugging Face dataset repository",
        description=(
            "Reads the token from HF_TOKEN or HUGGING_FACE_HUB_TOKEN only — never "
            "a flag, so it cannot land in a shell history or a CI log."
        ),
    )
    publish.add_argument("--export-dir", type=Path, default=None)
    publish.add_argument("--repo", default=None, help="defaults to $KANAL_HF_REPO")
    publish.add_argument("--private", action="store_true")
    publish.add_argument(
        "--dry-run",
        action="store_true",
        help="check the export is complete and report what would be pushed",
    )
    publish.set_defaults(func=_cmd_publish)

    status = sub.add_parser("status", help="show landing-zone and feed-registry state")
    status.set_defaults(func=_cmd_status)

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    try:
        return int(args.func(args))
    except KeyError as exc:  # unknown --source
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
