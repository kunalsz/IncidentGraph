PYTHON := $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)
export FALKORDB_HOST ?= localhost
export FALKORDB_PORT ?= 6380
PY := PYTHONPATH=backend:. $(PYTHON)

.PHONY: up down seed demo test surface eval

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
