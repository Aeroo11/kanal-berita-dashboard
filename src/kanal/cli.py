"""Command-line entry point.

kanal ingest [--source antara] [--dry-run]
kanal status
"""

from __future__ import annotations

import argparse
import logging
import sys

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
