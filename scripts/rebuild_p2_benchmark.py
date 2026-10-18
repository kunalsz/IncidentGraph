"""Acquire pinned RE2-OB telemetry and predeclare a replacement discovery split."""
import concurrent.futures
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_rcaeval import CASE_FILES, download_file, file_sha256, list_cases

REVISION = "afeacb11bcc94dadfd1c8f483ee4377b2b8b614e"
DEST = ROOT / "data/RE2-OB"
OUT = ROOT / "output/p2-rebuilt-20261018"
SEED = 20261018


def retry(action):
    for attempt in range(6):
        try:
            return action()
        except (httpx.HTTPError, ValueError):
            if attempt == 5:
                raise
            time.sleep(min(60, 5 * 2 ** attempt))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        cases = retry(lambda: list_cases(client, REVISION))
        assert len(cases) == 90
        discovery = []
        holdout = []
        rng = random.Random(SEED)
        # Each fault has 15 cases; preserve representation in both partitions.
        for i, fault in enumerate(("cpu", "mem", "disk", "socket", "delay", "loss")):
            group = [case for case in cases if case.rsplit("_", 2)[1] == fault]
            rng.shuffle(group)
            n = 9 if i < 2 else 8
            discovery.extend(group[:n])
            holdout.extend(group[n:])
        protocol = {"dataset_revision": REVISION, "seed": SEED,
                    "historical_records_recovered": False,
                    "scoring": "current repaired adapter; not historical scoring",
                    "discovery": discovery, "holdout": holdout,
                    "selection_rule": "highest discovery deployed AC@1 among nonbaseline arms; ties by pessimistic AC@1, then MRR, then declared arm order; choose baseline if no strict deployed AC@1 improvement",
                    "holdout_rule": "evaluate only baseline and selected arm; no retuning"}
        protocol_path = OUT / "protocol.json"
        if protocol_path.exists():
            assert json.loads(protocol_path.read_text()) == protocol
        else:
            protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
        for name, names in (("discovery", discovery), ("holdout", holdout)):
            (OUT / f"{name}-cases.txt").write_text("\n".join(names) + "\n")

    def fetch(case):
        with httpx.Client(follow_redirects=True, timeout=120) as client:
            def metadata():
                response = client.get(f"https://huggingface.co/api/datasets/phamquiluan/RCAEval/tree/{REVISION}/{case}")
                response.raise_for_status()
                return response.json()
            entries = {Path(item["path"]).name: item for item in retry(metadata)}
            size = sum(entries[name]["size"] for name in CASE_FILES)
            if shutil.disk_usage(DEST.parent).free < size + 500_000_000:
                raise RuntimeError("Insufficient disk headroom for safe acquisition")
            hashes = {}
            for name in CASE_FILES:
                entry = entries[name]
                # Xet CDN etags identify chunks, not the file's SHA256.
                expected = entry.get("lfs", {}).get("oid")
                retry(lambda: download_file(client, case, name, DEST, REVISION, expected))
                target = DEST / case / name
                if target.stat().st_size != entry["size"]:
                    raise ValueError(f"Size mismatch: {case}/{name}")
                if not expected:
                    data = target.read_bytes()
                    assert hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == entry["oid"]
                hashes[name] = file_sha256(target)
            print(case, "verified", flush=True)
            return case, hashes

    DEST.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        hashes = dict(pool.map(fetch, cases))
    meta = {"dataset": "RE2-OB", "revision": REVISION, "actual_count": len(cases),
            "selected": discovery + holdout, "selection_order": discovery + holdout,
            "checksums": hashes, "files": list(CASE_FILES), "missing_modalities": [],
            "source": "https://huggingface.co/datasets/phamquiluan/RCAEval", "seed": SEED}
    (DEST / "_subset.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("Acquisition complete", flush=True)


if __name__ == "__main__":
    main()
