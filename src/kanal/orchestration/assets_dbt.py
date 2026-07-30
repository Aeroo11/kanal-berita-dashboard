"""dbt models, pulled into the same asset graph.

This is the main reason Dagster is here rather than a Makefile. `dagster-dbt`
turns every dbt model into an asset and every dbt test into an asset check, so
the lineage view spans ingestion *and* transformation in one graph: feeds →
landing zone → raw_articles → staging → intermediate → marts.

Two separate DAGs — one in a Makefile, one inside dbt — would each be correct and
neither would answer "where did this number come from?" without a human joining
them by hand.
"""

# No `from __future__ import annotations` in this module, deliberately. Dagster
# inspects the `context` parameter's annotation at decoration time to decide what
# to pass in; the future import turns annotations into strings and Dagster then
# rejects the asset with "Cannot annotate `context` parameter with type
# AssetExecutionContext". Every other module in the project keeps the import.

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dagster import AssetExecutionContext, AssetKey
from dagster_dbt import DagsterDbtTranslator, DbtCliResource, DbtProject, dbt_assets

DBT_DIR = Path(__file__).resolve().parents[3] / "dbt"

dbt_project = DbtProject(
    project_dir=DBT_DIR,
    # Checked in alongside the project. There is no secret in it — the warehouse
    # is a local DuckDB file — and a profile living only on one laptop makes the
    # project unrunnable for anyone else, CI included.
    profiles_dir=DBT_DIR,
)

# Builds the manifest if it is missing or stale. Without this, a fresh clone has
# no manifest and Dagster cannot enumerate the dbt assets at all.
dbt_project.prepare_if_dev()

dbt_resource = DbtCliResource(project_dir=dbt_project)


class KanalDbtTranslator(DagsterDbtTranslator):
    """Make dbt's source keys match the Python assets they actually refer to.

    Without this the graph silently splits in two. dbt declares its source as
    `kanal.raw_articles`, which dagster-dbt turns into the asset key
    `kanal/raw_articles` — while the Python asset that writes that table is
    keyed `raw_articles`. Both nodes then exist, nothing connects them, and the
    lineage view shows two disconnected clusters that each look fine.

    That failure is quiet in the worst way: every asset materialises, every check
    passes, and the one thing Dagster was added for — being able to trace a
    number in a mart back to the feed it came from — is the thing that no longer
    works.
    """

    def get_asset_key(self, dbt_resource_props: Mapping[str, Any]) -> AssetKey:
        if dbt_resource_props["resource_type"] == "source":
            # Sources are tables the Python layer owns. Key them by name alone,
            # matching the asset that writes them.
            return AssetKey(dbt_resource_props["name"])
        return super().get_asset_key(dbt_resource_props)

    def get_group_name(self, dbt_resource_props: Mapping[str, Any]) -> str | None:
        """Group by dbt layer, so the lineage view reads as staging → marts.

        dbt's own folder structure already encodes the layering; surfacing it as
        Dagster groups means the graph is legible without clicking anything.
        """
        path = dbt_resource_props.get("fqn", [])
        for layer in ("staging", "intermediate", "marts"):
            if layer in path:
                return layer
        if dbt_resource_props["resource_type"] == "seed":
            return "seeds"
        return super().get_group_name(dbt_resource_props)


@dbt_assets(
    manifest=dbt_project.manifest_path,
    dagster_dbt_translator=KanalDbtTranslator(),
)
def kanal_dbt_assets(context: AssetExecutionContext, dbt: DbtCliResource):  # type: ignore[no-untyped-def]
    """Every dbt model, seed and test as Dagster assets and checks.

    `build` rather than `run`: it runs seeds, models and tests in dependency
    order, so the taxonomy seed lands before the models that join it and a failed
    contract stops its dependents instead of letting them build on bad data.
    """
    yield from dbt.cli(["build"], context=context).stream()
