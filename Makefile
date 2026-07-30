# Operational commands. The ones that matter during an incident are at the top.

REGISTRY ?= data/registry

.PHONY: rollback
rollback:  ## Swap champion and previous. The API follows within 60s, no redeploy.
	@uv run kanal rollback --registry $(REGISTRY)

.PHONY: champion
champion:  ## What is serving right now.
	@uv run kanal champion --registry $(REGISTRY)

.PHONY: decisions
decisions:  ## The promotion log, including what was turned down.
	@uv run kanal decisions --registry $(REGISTRY)

.PHONY: serve
serve:  ## Run the API locally on :8000, with Swagger at /docs.
	@KANAL_REGISTRY=$(REGISTRY) uv run uvicorn kanal.serving.api:app --reload --port 8000

.PHONY: ingest
ingest:  ## One ingestion cycle. Run twice; the second adds nothing.
	@uv run kanal ingest

.PHONY: build
build:  ## Load the landing zone into DuckDB and rebuild the dbt models.
	@uv run kanal load
	@cd dbt && uv run dbt build

.PHONY: test
test:  ## The full suite.
	@uv run pytest

.PHONY: check
check:  ## Everything CI checks, in the order CI checks it.
	@uv run ruff format --check .
	@uv run ruff check .
	@uv run mypy
	@uv run pytest

.PHONY: docker
docker:  ## Build the serving image.
	@docker build -t kanal-api .

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
