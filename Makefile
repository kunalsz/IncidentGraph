PYTHON := $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)
export FALKORDB_HOST ?= localhost
export FALKORDB_PORT ?= 6380
PY := PYTHONPATH=backend:. $(PYTHON)

.PHONY: up down seed demo test surface eval eval-rcaeval eval-rcaeval-agent

up:
	docker compose up -d

down:
	docker compose down

seed:
	$(PY) scripts/seed_demo.py

demo:
	$(PY) scripts/demo_inject.py --fixture payment_failure --run-p0

test:
	$(PY) -m pytest tests -q

surface:
	$(PY) scripts/measure_tool_surface.py

eval:
	$(PY) scripts/eval.py --fixtures --rank-only

# External benchmark (requires `python scripts/fetch_rcaeval.py` first)
eval-rcaeval:
	$(PY) scripts/eval.py --rcaeval data/RE2-OB

# External benchmark with the agent (P3-P5, live LLM; needs LLM_API_KEY)
eval-rcaeval-agent:
	$(PY) scripts/eval.py --rcaeval data/RE2-OB --with-agent --report output/rcaeval-agent
