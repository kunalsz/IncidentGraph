"""Download a pinned RCAEval suite (re2ss/re2tt/...) with per-file verification.

Generalises the OB-only fetcher: case prefix is a parameter, files are verified
against the dataset's LFS oids and sizes, and a predeclared subset rule (first N
cases per fault, lexicographic) keeps the download bounded.
"""
import argparse
import concurrent.futures
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_rcaeval import CASE_FILES, download_file, file_sha256

HF = "phamquiluan/RCAEval"
REVISION = "afeacb11bcc94dadfd1c8f483ee4377b2b8b614e"


def retry(action, attempts=7):
    """Retry transient failures, including the API's 429 rate limit.

    The dataset API allows 500 requests per 300 s, and one case costs several
    requests, so backoff has to be able to span a full window.
    """
    for attempt in range(attempts):
        try:
            return action()
        except (httpx.HTTPError, ValueError):
            if attempt == attempts - 1:
                raise
            time.sleep(min(120, 10 * 2 ** attempt))


def _get(client, url, params=None):
    """One request that raises inside the retried call, 429 included."""
    response = client.get(url, params=params)
    response.raise_for_status()
    return response


def list_suite(client, prefix):
    """Top-level case directories for one suite, across all pages.

    Non-recursive listing: one entry per case directory, so the page count is
    small and no partial page can be mistaken for the whole suite.
    """
    url = f"https://huggingface.co/api/datasets/{HF}/tree/{REVISION}"
    found = set()
    while url:
        response = retry(lambda: _get(client, url, {"limit": "1000"}))
        if not isinstance(response.json(), list):
            raise ValueError("unexpected listing page")
        found.update(row["path"] for row in response.json()
                     if row.get("type") == "directory" and row["path"].startswith(prefix))
        url = response.links.get("next", {}).get("url")
    return sorted(found)


def per_fault(names, per_fault_limit):
    """First N case names per fault class, lexicographic - predeclared rule."""
    groups = {}
    for name in sorted(names):
        groups.setdefault(name.rsplit("_", 2)[1], []).append(name)
    return [name for fault in sorted(groups) for name in groups[fault][:per_fault_limit]]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", required=True, help="case prefix, e.g. re2ss_")
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--per-fault", type=int, default=10)
    args = parser.parse_args()
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        names = retry(lambda: list_suite(client, args.suite))
        selected = per_fault(names, args.per_fault)
        print(f"{args.suite}: {len(names)} cases available, selecting {len(selected)}", flush=True)
        args.dest.mkdir(parents=True, exist_ok=True)

        def fetch(case):
            with httpx.Client(follow_redirects=True, timeout=120) as client:
                rows = retry(lambda: _get(
                    client, f"https://huggingface.co/api/datasets/{HF}/tree/{REVISION}/{case}")).json()
                entries = {Path(row["path"]).name: row for row in rows}
                # Suites differ: re2ss ships metrics+logs only. Fetch exactly
                # what exists and record the absent modalities rather than
                # failing (or silently degrading) at load time.
                present = [name for name in CASE_FILES if name in entries]
                size = sum(entries[name]["size"] for name in present)
                if shutil.disk_usage(args.dest.parent).free < size + 400_000_000:
                    raise RuntimeError("insufficient disk headroom")
                for name in present:
                    entry = entries[name]
                    expected = entry.get("lfs", {}).get("oid")
                    retry(lambda: download_file(client, case, name, args.dest, REVISION, expected))
                    if (args.dest / case / name).stat().st_size != entry["size"]:
                        raise ValueError(f"size mismatch: {case}/{name}")
            print(case, "verified", ",".join(present), flush=True)
            return present
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            inventory = list(pool.map(fetch, selected))
    files_present = sorted({name for row in inventory for name in row})
    meta = {"dataset": "RCAEval", "suite": args.suite, "revision": REVISION,
            "selection": f"first {args.per_fault} per fault class, lexicographic",
            "available": len(names), "selected": selected,
            "files": files_present,
            "missing_modalities": [name for name in CASE_FILES if name not in files_present],
            "checksums": {case: {name: file_sha256(args.dest / case / name)
                                 for name in files_present if (args.dest / case / name).is_file()}
                          for case in selected},
            "source": f"https://huggingface.co/datasets/{HF}"}
    (args.dest / "_subset.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", args.dest / "_subset.json")


if __name__ == "__main__":
    main()
