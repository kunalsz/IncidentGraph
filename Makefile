PYTHON := $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)
# Docker Compose v2 ships as a `docker compose` plugin; v1 as `docker-compose`.
DC := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo docker-compose)
export FALKORDB_HOST ?= localhost
export FALKORDB_PORT ?= 6380
# The UI service runs against its own database (its runs are real investigations
# and must not share the demo graph). Override with INCIDENTGRAPH_UI_DB_PORT.
UI_DB_PORT ?= 6381
UI_DB_NAME ?= incidentgraph-ui-db
export INCIDENTGRAPH_UI_DB_HOST ?= 127.0.0.1
export INCIDENTGRAPH_UI_DB_PORT ?= $(UI_DB_PORT)
PY := PYTHONPATH=backend:. $(PYTHON)

.PHONY: up down ui-db ui-db-down seed demo api web test surface eval eval-rcaeval eval-rcaeval-agent

# Compose first, then fall back to a plain container: old docker-compose v1
# releases fail against recent Docker APIs ("Not supported URL scheme
# http+docker"), and a single FalkorDB container is all this target needs.
up:
	@$(DC) up -d 2>/dev/null || { \
		docker start incidentgraph-falkordb >/dev/null 2>&1 || \
		docker run -d --name incidentgraph-falkordb -p 6380:6379 falkordb/falkordb:latest >/dev/null; \
		echo "FalkorDB on localhost:6380 (container incidentgraph-falkordb)"; }

down:
	@$(DC) down 2>/dev/null || docker stop incidentgraph-falkordb >/dev/null 2>&1 || true

# FalkorDB for the web service, on $(UI_DB_PORT), separate from the demo graph.
ui-db:
	@docker start $(UI_DB_NAME) >/dev/null 2>&1 || docker run -d --name $(UI_DB_NAME) -p $(UI_DB_PORT):6379 falkordb/falkordb:latest
	@echo "FalkorDB for the UI on 127.0.0.1:$(UI_DB_PORT) ($(UI_DB_NAME))"

ui-db-down:
	@docker stop $(UI_DB_NAME) >/dev/null 2>&1 || true

# API + UI (run in two terminals; see README "Web demo")
api:
	$(PY) -m uvicorn server.app.main:app --host 127.0.0.1 --port 8000

web:
	cd web && npm install --no-fund --no-audit && npm run dev

seed:
	$(PY) scripts/seed_demo.py

# --seed-demo is required: the fixture's alert only exists after the fixture is
# injected, so P0 cannot resolve an anchor on a freshly seeded graph without it.
demo:
	$(PY) scripts/demo_inject.py --fixture payment_failure --seed-demo --run-p0

# Needs a FalkorDB on $(FALKORDB_HOST):$(FALKORDB_PORT) (`make up`, or `make ui-db`).
# Unit only: `$(PY) -m pytest tests/unit -q`.
test:
	$(PY) -m pytest tests -q

surface:
	$(PY) scripts/measure_tool_surface.py

eval:
	$(PY) scripts/eval.py --fixtures --rank-only

# External benchmark (requires `python scripts/fetch_rcaeval.py` first)
eval-rcaeval:
	$(PY) scripts/eval.py --rcaeval data/RE2-OB

# Live-agent evaluation is intentionally not a CLI flag (the CLI refuses to
# reach a provider implicitly). Drive it programmatically with an explicit
# client factory, or through the web service; see README "Make targets".
eval-rcaeval-agent:
	@echo "eval.py refuses implicit live providers by design."; \
	echo "Use rcaeval_runner.run_case(..., llm_factory=...) or the web service."; exit 1
