#!/usr/bin/env python
"""Fetch RCAEval RE2-OB cases from HuggingFace into data/RE2-OB/.

The HuggingFace parquet conversion (phamquiluan/RCAEval) is the layout
RCAEval's own readers prefer (`resolve_case_file`), and it allows per-case
downloads — no multi-GB Zenodo archive needed for a subset run.

Layout produced:
  data/RE2-OB/re2ob_<service>_<fault>_<instance>/{inject_time.txt,
      metrics.parquet, logs.parquet, traces.parquet}
  data/RE2-OB/_subset.json    # the selected subset, for reproducibility

Case directories are named `re2ob_<service>_<fault>_<instance>`; the name is
ground truth and is consumed only by the evaluator (`rcaeval_runner.py`),
never by the adapter that feeds the engine.

Usage:
  .venv/bin/python scripts/fetch_rcaeval.py --subset 20
  .venv/bin/python scripts/fetch_rcaeval.py --all
  .venv/bin/python scripts/fetch_rcaeval.py --cases re2ob_checkoutservice_cpu_1,...
"""

import argparse
import hashlib
import uuid
import json
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = REPO_ROOT / "data" / "RE2-OB"

HF_DATASET = "phamquiluan/RCAEval"
HF_API = f"https://huggingface.co/api/datasets/{HF_DATASET}/tree/main"
HF_RESOLVE = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main"

CASE_PREFIX = "re2ob_"
CASE_FILES = ("inject_time.txt", "metrics.parquet", "logs.parquet", "traces.parquet")


def _client() -> httpx.Client:
    return httpx.Client(follow_redirects=True, timeout=120.0)


def list_cases(client: httpx.Client, revision: str = "main") -> list[str]:
    """Follow HF pagination; revision is resolved to a commit by main()."""
    url = f"https://huggingface.co/api/datasets/{HF_DATASET}/tree/{revision}"
    found = set()
    while url:
        resp = client.get(url, params={"recursive": "false", "limit": "1000"})
        resp.raise_for_status()
        found.update(entry["path"] for entry in resp.json()
                     if entry["type"] == "directory" and entry["path"].startswith(CASE_PREFIX))
        url = resp.links.get("next", {}).get("url")
    return sorted(found)


def parse_case(case: str) -> tuple[str, str, str]:
    """`re2ob_checkoutservice_cpu_1` -> ('checkoutservice', 'cpu', '1')."""
    parts = case.replace(CASE_PREFIX, "", 1).rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(f"unexpected case name: {case}")
    return parts[0], parts[1], parts[2]


def select_subset(cases: list[str], n: int, seed: int = 0) -> list[str]:
    """Nested exact-count prefixes: fault/service rounds, then next instances."""
    if n <= 0:
        raise ValueError("subset must be positive")
    groups = {}
    for case in sorted(set(cases)):
        service, fault, instance = parse_case(case)
        groups.setdefault((service, fault), []).append((int(instance), case))
    if not groups:
        return []
    services = sorted({s for s, f in groups})
    faults = sorted({f for s, f in groups})
    # Diagonal rotations cover faults and services early, and every group once.
    order = []
    for round_index in range(len(services)):
        for index, fault in enumerate(faults):
            service = services[(index + round_index + seed) % len(services)]
            if (service, fault) in groups:
                order.append((service, fault))
    queues = {key: [case for _,case in sorted(value)] for key,value in groups.items()}
    selected = []
    while any(queues.values()):
        for key in order:
            if queues[key]:
                selected.append(queues[key].pop(0))
    return selected[:min(n, len(selected))]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(client: httpx.Client, case: str, name: str, dest: Path,
                  revision: str = "main", expected_sha256: str | None = None) -> str:
    """Atomic verified download; SHA256 LFS etag or an explicit trusted digest.

    Non-LFS Git blob etags are validated using Git's SHA1 blob framing. Reuse
    always rehashes bytes. A length match alone is never sufficient.
    """
    url = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/{revision}/{case}/{name}"
    target = dest / case / name
    head = client.head(url)
    head.raise_for_status()
    etag = head.headers.get("x-linked-etag", head.headers.get("etag", "")).strip('"')
    expected = int(head.headers.get("x-linked-size", head.headers.get("content-length", 0)) or 0)
    expected_sha256 = expected_sha256 or (etag if len(etag) == 64 else None)
    def valid(path):
        if expected and path.stat().st_size != expected:
            return False
        if expected_sha256:
            return file_sha256(path) == expected_sha256
        if len(etag) == 40:
            data = path.read_bytes()
            return hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest() == etag
        return False
    if target.exists() and valid(target):
        return "skipped"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + f".{uuid.uuid4().hex}.part")
    try:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as handle:
                for chunk in resp.iter_bytes(1024 * 1024):
                    handle.write(chunk)
        if not valid(tmp):
            raise ValueError(f"missing or mismatched trusted checksum: {case}/{name}")
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
    return "downloaded"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--subset", type=int, default=20,
                        help="number of cases to fetch (default 20)")
    parser.add_argument("--all", action="store_true", help="fetch all 90 cases")
    parser.add_argument("--cases", type=str, default=None,
                        help="explicit comma-separated case names (overrides subset)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-traces", action="store_true",
                        help="skip traces.parquet (~10MB/case); severity degrades honestly")
    parser.add_argument("--revision", default="main", help="HF revision resolved once to immutable commit")
    args = parser.parse_args()
    if args.subset <= 0:
        parser.error("subset must be positive")

    with _client() as client:
        revision_response = client.get(f"https://huggingface.co/api/datasets/{HF_DATASET}/revision/{args.revision}")
        revision_response.raise_for_status()
        revision = revision_response.json()["sha"]
        all_cases = list_cases(client, revision)
        if not all_cases:
            print("no RE2-OB cases found upstream", file=sys.stderr)
            return 1

        if args.cases:
            requested = [c.strip() for c in args.cases.split(",") if c.strip()]
            unknown = [c for c in requested if c not in all_cases]
            if unknown:
                print(f"unknown cases: {unknown}", file=sys.stderr)
                return 1
            if len(set(requested)) != len(requested):
                parser.error("duplicate requested cases")
            selected = requested
        elif args.all:
            selected = select_subset(all_cases, len(all_cases), args.seed)
        else:
            selected = select_subset(all_cases, args.subset, args.seed)

        files = [f for f in CASE_FILES if not (args.no_traces and f == "traces.parquet")]
        print(f"RE2-OB cases available: {len(all_cases)}; fetching {len(selected)}")
        for case in selected:
            outcomes = {}
            for name in files:
                outcomes[name] = download_file(client, case, name, args.dest, revision)
            status = ", ".join(f"{k}={'skip' if v == 'skipped' else 'get'}"
                               for k, v in outcomes.items())
            print(f"  {case}: {status}")

    meta = {
        "dataset": "RE2-OB",
        "revision": revision, "requested_count": len(requested) if args.cases else (len(all_cases) if args.all else args.subset),
        "actual_count": len(selected), "selection_order": selected,
        "checksums": {case: {name: file_sha256(args.dest/case/name) for name in files} for case in selected},
        "missing_modalities": [name for name in CASE_FILES if name not in files],
        "source": f"https://huggingface.co/datasets/{HF_DATASET}",
        "available": len(all_cases),
        "selected": selected,
        "files": files,
        "subset": None if args.cases or args.all else args.subset,
        "seed": args.seed,
    }
    meta_path = args.dest / "_subset.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_meta = meta_path.with_suffix(".json.part")
    tmp_meta.write_text(json.dumps(meta, indent=2) + "\n")
    tmp_meta.replace(meta_path)
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
