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
import time
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


def _retry(action, attempts=7):
    """Retry only TRANSIENT failures, with backoff that can span a rate window.

    Transport errors, 429 and 5xx are transient; every other 4xx is a real
    answer (a missing file, a bad revision) and must fail immediately. Retrying
    those cost a measured 6.5 minutes per missing file, because
    `HTTPStatusError` is itself an `HTTPError`.
    """
    for attempt in range(attempts):
        try:
            return action()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status != 429 and status < 500:
                raise
            if attempt == attempts - 1:
                raise
            time.sleep(min(120, 10 * 2 ** attempt))
        except httpx.TransportError:
            if attempt == attempts - 1:
                raise
            time.sleep(min(120, 10 * 2 ** attempt))


def get_with_retry(client, url, params=None):
    return _retry(lambda: _ok(client.get(url, params=params)))


def head_with_retry(client, url):
    return _retry(lambda: _ok(client.head(url)))


def stream_with_retry(client, url):
    """Open a streaming response, retrying the status before any bytes are read."""
    def open_stream():
        context = client.stream("GET", url)
        response = context.__enter__()
        try:
            return _ok(response), context
        except BaseException:
            context.__exit__(None, None, None)
            raise
    response, context = _retry(open_stream)
    return _StreamGuard(context, response)


class _StreamGuard:
    """Closes the streaming context when its `with` block exits or fails."""
    def __init__(self, context, response):
        self._context, self.response = context, response

    def __enter__(self):
        return self.response

    def __exit__(self, *exc):
        return self._context.__exit__(*exc)


def _ok(response):
    if response.status_code == 429:
        # Honour Retry-After when the server sends it, then let backoff run.
        raw = response.headers.get("retry-after")
        try:
            delay = float(raw) if raw else None
        except (TypeError, ValueError):
            delay = None
        if delay:
            time.sleep(min(delay, 120))
    response.raise_for_status()
    return response


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


def git_blob_sha1(data: bytes) -> str:
    """Git's own object hash for a blob: sha1("blob <len>\0" + content)."""
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def case_files(client: httpx.Client, case: str, revision: str) -> dict:
    """File metadata for one case: size, plus the digest that can verify it.

    Large files are stored as LFS pointers whose `lfs.oid` IS the content's
    SHA256; small ones are ordinary Git blobs identified by `oid`. The resolve
    endpoint's ETag is NOT either of those on HuggingFace's Xet backend (it is a
    chunk hash), which is why a download verified from it fails - see
    download_file.
    """
    response = get_with_retry(
        client, f"https://huggingface.co/api/datasets/{HF_DATASET}/tree/{revision}/{case}")
    entries = {}
    for row in response.json():
        name = Path(row["path"]).name
        lfs = (row.get("lfs") or {}).get("oid")
        entries[name] = {"size": row.get("size"),
                         "sha256": lfs if isinstance(lfs, str) and len(lfs) == 64 else None,
                         "git_oid": None if lfs else row.get("oid")}
    return entries


def download_file(client: httpx.Client, case: str, name: str, dest: Path,
                  revision: str = "main", expected_sha256: str | None = None, *,
                  expected_size: int | None = None,
                  expected_blob_sha1: str | None = None) -> str:
    """Atomic verified download of one case file, or "skipped" when already valid.

    Verification, in order of preference: an explicit SHA256 (LFS content hash),
    Git's SHA1 blob framing, or - only when neither was supplied - the ETag from
    the resolve endpoint, which is a fallback rather than a guarantee. A length
    match alone is never sufficient, and a file that fails verification is
    removed rather than kept.
    """
    url = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/{revision}/{case}/{name}"
    target = dest / case / name
    if not (expected_sha256 or expected_blob_sha1):
        head = head_with_retry(client, url)
        etag = head.headers.get("x-linked-etag", head.headers.get("etag", "")).strip('"')
        expected_size = int(head.headers.get("x-linked-size",
                                             head.headers.get("content-length", 0)) or 0)
        if len(etag) == 64:
            expected_sha256 = etag
        elif len(etag) == 40:
            expected_blob_sha1 = etag

    def digest_ok(path):
        if expected_size is not None and path.stat().st_size != expected_size:
            return False
        if expected_sha256:
            return file_sha256(path) == expected_sha256
        if expected_blob_sha1:
            return git_blob_sha1(path.read_bytes()) == expected_blob_sha1
        raise ValueError(f"no trusted digest for {case}/{name}: refusing to accept a "
                         "download verified only by length")

    if target.exists() and digest_ok(target):
        return "skipped"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + f".{uuid.uuid4().hex}.part")
    try:
        with stream_with_retry(client, url) as resp:
            with tmp.open("wb") as handle:
                for chunk in resp.iter_bytes(1024 * 1024):
                    handle.write(chunk)
        if not digest_ok(tmp):
            raise ValueError(f"checksum mismatch: {case}/{name} "
                             f"(expected {expected_sha256 or expected_blob_sha1})")
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
        revision_response = get_with_retry(
            client, f"https://huggingface.co/api/datasets/{HF_DATASET}/revision/{args.revision}")
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

        wanted = [f for f in CASE_FILES if not (args.no_traces and f == "traces.parquet")]
        print(f"RE2-OB cases available: {len(all_cases)}; fetching {len(selected)}")
        present, missing_modalities = set(), set()
        for case in selected:
            # Per-file digest from the dataset's own metadata: LFS blobs are
            # verified by content SHA256, small files by Git's blob hash. This
            # costs one API call per case and removes a HEAD per file.
            metadata = case_files(client, case, revision)
            outcomes = {}
            for name in wanted:
                entry = metadata.get(name)
                if entry is None:
                    missing_modalities.add(name)
                    outcomes[name] = "absent upstream"
                    continue
                present.add(name)
                outcomes[name] = download_file(
                    client, case, name, args.dest, revision,
                    expected_sha256=entry["sha256"], expected_size=entry["size"],
                    expected_blob_sha1=entry["git_oid"])
            status = ", ".join(f"{k}={'skip' if v == 'skipped' else v}" for k, v in outcomes.items())
            print(f"  {case}: {status}", flush=True)

    meta = {
        "dataset": "RE2-OB",
        "revision": revision, "requested_count": len(requested) if args.cases else (len(all_cases) if args.all else args.subset),
        "actual_count": len(selected), "selection_order": selected,
        "checksums": {case: {name: file_sha256(args.dest/case/name) for name in sorted(present)
                             if (args.dest/case/name).is_file()} for case in selected},
        "missing_modalities": sorted(set(CASE_FILES) - present),
        "source": f"https://huggingface.co/datasets/{HF_DATASET}",
        "available": len(all_cases),
        "selected": selected,
        "files": sorted(present),
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
