"""Configuration for the live pipeline service.

The service owns a disposable database and a bounded run budget. It never
imports or edits the investigation engine; it only calls its public entry
points, so the agent codebase stays untouched.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "RE2-OB"
RUNS_DIR = ROOT / "output" / "ui-runs"
WEB_DIST = ROOT / "web" / "dist"

# Disposable database dedicated to this UI. Note this is deliberately not the
# compose instance on 6380: the UI drives real pipeline runs, and those must not
# write into the shared demo database. Point it elsewhere with these variables.
DB_HOST = os.environ.get("INCIDENTGRAPH_UI_DB_HOST", "127.0.0.1")
DB_PORT = int(os.environ.get("INCIDENTGRAPH_UI_DB_PORT", "6381"))
CONTAINER_NAME = "incidentgraph-ui-db"

# Branch parallelism for live runs: enough to finish in ~90s, low enough to
# stay well inside provider rate limits.
PARALLELISM = 4

# Bounded per-run budget. The ceilings keep a live run finite; the headroom
# above the observed ~34 calls / ~153k tokens keeps P6's live read path
# available instead of silently downgrading the report to INCONCLUSIVE.
BUDGET_CAPS = {
    "max_llm_calls": 75,
    "max_tokens": 220_000,
    "max_wall_s": 540,
    "max_nodes_visited": 200,
    "max_hypotheses": 40,
}

# RCAEval case directories follow the benchmark's naming contract.
CASE_PATTERN = re.compile(r"^(re2ob_[A-Za-z0-9-]+_(?:cpu|mem|disk|socket|delay|loss)_[0-9]+)$")

# Ten cases spanning every injected service and fault class, including the ones
# that expose anchor confusion, metric-scoring blind spots and near misses.
FEATURED_CASES: tuple[str, ...] = (
    "re2ob_checkoutservice_cpu_1",
    "re2ob_checkoutservice_delay_1",
    "re2ob_currencyservice_cpu_1",
    "re2ob_currencyservice_socket_3",
    "re2ob_emailservice_loss_1",
    "re2ob_emailservice_delay_1",
    "re2ob_productcatalogservice_disk_2",
    "re2ob_productcatalogservice_mem_1",
    "re2ob_recommendationservice_disk_1",
    "re2ob_recommendationservice_cpu_2",
)

STAGES: tuple[tuple[str, str], ...] = (
    ("p0", "resolve"),
    ("p1", "scope"),
    ("p2", "rank"),
    ("p3", "investigate"),
    ("p4", "validate"),
    ("p5", "select"),
    ("p6", "report"),
)

# Evaluator aliasing: faults are scored against these indicator families.
FAULT_METRIC = {
    "delay": "latency",
    "loss": "latency",
    "disk": "diskio",
    "socket": "socket",
    "cpu": "cpu",
    "mem": "mem",
}

CORS_ORIGINS = ["http://127.0.0.1:5173", "http://localhost:5173"]

# RCAEval ships its telemetry on HuggingFace (the GitHub repo holds code only),
# so a case links to its own folder in the dataset at the revision the local
# copy was fetched from.
DATASET_REPO = "https://huggingface.co/datasets/phamquiluan/RCAEval"
DATASET_REVISION = "afeacb11bcc94dadfd1c8f483ee4377b2b8b614e"


def case_source_url(case_id: str) -> str:
    return f"{DATASET_REPO}/tree/{DATASET_REVISION}/{case_id}"


def parse_case(case_id: str) -> dict[str, str]:
    """`re2ob_<service>_<fault>_<instance>` → display fields."""
    parts = case_id.removeprefix("re2ob_").rsplit("_", 2)
    service, fault, instance = parts if len(parts) == 3 else (case_id, "?", "?")
    return {"case": case_id, "service": service, "fault": fault, "instance": instance}


def available_cases() -> list[dict[str, str]]:
    """Featured cases that are actually present in the local dataset."""
    found = []
    for case_id in FEATURED_CASES:
        if (DATA_DIR / case_id).is_dir():
            found.append({**parse_case(case_id), "source_url": case_source_url(case_id)})
    return found
