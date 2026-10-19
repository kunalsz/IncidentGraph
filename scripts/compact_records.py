"""Shrink archived run records to what analyses actually replay, keeping proof.

Per case record, ~95% of the bytes are (a) `handoffs`, which duplicate the
`artifacts` payloads for integrity hand-offs, and (b) scope `nodes` of evidence
type (thousands of spans/log lines) plus their OBSERVED_ON edges. Every offline
analysis - policy replay, the sampler audit, the anchor-prior comparison - reads
instead: the ranking's per-candidate components, the scope's services,
DEPENDS_ON edges, anchor and window, and the agent artifacts.

Nothing is deleted silently: each compacted record records what was removed
(counts), the original byte size and the original SHA256, so an archive can
prove what it used to contain. Records are only rewritten when smaller.
"""
import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact(record):
    """Return (compacted, removal report) or (None, report) when nothing to do."""
    removed = {}
    out = dict(record)
    if "handoffs" in out:
        removed["handoffs"] = {"bytes": len(json.dumps(out["handoffs"])),
                               "stages": sorted(out["handoffs"])}
        out.pop("handoffs")
    if "window_frame" in out:
        removed["window_frame"] = {"bytes": len(json.dumps(out["window_frame"]))}
        out.pop("window_frame")
    artifacts = out.get("artifacts")
    if isinstance(artifacts, dict) and isinstance(artifacts.get("scope"), dict):
        scope = dict(artifacts["scope"])
        nodes = scope.get("nodes") or []
        edges = scope.get("edges") or []
        kept_nodes = [n for n in nodes if n.get("type") == "Service"]
        kept_edges = [e for e in edges if e.get("type") != "OBSERVED_ON"]
        if len(kept_nodes) != len(nodes) or len(kept_edges) != len(edges):
            removed["scope_evidence_nodes"] = {
                "bytes": len(json.dumps(nodes)) - len(json.dumps(kept_nodes)),
                "nodes_removed": len(nodes) - len(kept_nodes),
                "by_type": {t: sum(1 for n in nodes if n.get("type") == t)
                            for t in sorted({n.get("type") for n in nodes} - {"Service"})}}
            removed["scope_observation_edges"] = {
                "bytes": len(json.dumps(edges)) - len(json.dumps(kept_edges)),
                "edges_removed": len(edges) - len(kept_edges)}
            scope["nodes"], scope["edges"] = kept_nodes, kept_edges
            artifacts = dict(artifacts, scope=scope)
            out["artifacts"] = artifacts
    if not removed:
        return None, removed
    return out, removed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dirs", nargs="+", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    total_before = total_after = files = 0
    for root in args.dirs:
        for path in sorted(root.rglob("*.record.json")):
            before = path.stat().st_size
            record = json.loads(path.read_text())
            compacted, removed = compact(record)
            if compacted is None:
                continue
            compacted["_compaction"] = {
                "reason": "evidence nodes and hand-off payloads removed; analyses read "
                          "ranking components, scope services/edges and agent artifacts",
                "original_bytes": before, "original_sha256": sha256(path), "removed": removed}
            payload = json.dumps(compacted, indent=2, allow_nan=False) + "\n"
            if len(payload) >= before:
                continue
            files += 1
            total_before += before
            total_after += len(payload)
            if not args.dry_run:
                tmp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
                tmp.write_text(payload)
                os.replace(tmp, path)
    print(f"{files} records {'would be ' if args.dry_run else ''}compacted: "
          f"{total_before/1048576:.0f} MB -> {total_after/1048576:.0f} MB "
          f"(freed {(total_before-total_after)/1048576:.0f} MB)")


if __name__ == "__main__":
    main()
