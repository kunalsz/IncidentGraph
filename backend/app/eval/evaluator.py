"""RCAEval scoring semantics, ported (MIT).

Source: https://github.com/phamquiluan/RCAEval (MIT License)
  - RCAEval/benchmark/evaluation.py  -> Evaluator (accuracy / chance / lift /
    retrieval / rerank, fine-grained and service-level)
  - RCAEval/classes/graph.py::Node   -> Node (entity + metric, value equality)
  - main.py                          -> rank-string parsing, dedup-preserving
    service ranking, per-fault-class evaluation buckets, candidate counting
  - RCAEval/io/time_series.py        -> drop_time / drop_constant used for the
    candidate-count denominator

We re-derive these ~200 lines in our own style (no vendoring — master plan
§13) and add one metric RCAEval does not report: service-level MRR.

Everything ground-truth-facing lives here; the adapter/engine never imports
this module.
"""

from __future__ import annotations

from typing import List, Sequence


class Node:
    """The element of a ranking: (entity, metric). Equality is by value pair."""

    def __init__(self, entity: str, metric: str):
        self._entity = entity
        self._metric = metric

    @property
    def entity(self) -> str:
        return self._entity

    @property
    def metric(self) -> str:
        return self._metric

    def asdict(self) -> dict:
        return {"entity": self._entity, "metric": self._metric}

    def __eq__(self, obj: object) -> bool:
        if isinstance(obj, Node):
            return self.entity == obj.entity and self.metric == obj.metric
        return False

    def __hash__(self) -> int:
        return hash((self._entity, self._metric))

    def __repr__(self) -> str:
        return f"Node{(self._entity, self._metric)}"


# ---------------------------------------------------------------------------
# Rank-string parsing (RCAEval main.py rules, verbatim semantics)
# ---------------------------------------------------------------------------

def service_rank_entities(rank_strings: Sequence[str]) -> List[str]:
    """Service-level entity list: first `_` token, `-db` stripped, dedup.

    Dedup preserves first occurrence (main.py keeps the best-ranked
    occurrence of each entity).
    """
    entities: List[str] = []
    for rank in rank_strings:
        entity = str(rank).split("_")[0].replace("-db", "")
        if entity not in entities:
            entities.append(entity)
    return entities


def fine_ranks(rank_strings: Sequence[str]) -> List[Node]:
    """Fine-grained (entity, metric) list: first two `_` tokens, no dedup."""
    return list(dict.fromkeys(
        Node(*x.split("_", 1)) for x in rank_strings if "_" in x
    ))


def service_ranks(rank_strings: Sequence[str]) -> List[Node]:
    """Service-level Node list with main.py's dedup-preserving order."""
    return [Node(entity, "unknown") for entity in service_rank_entities(rank_strings)]


# Ground-truth metric family per injected fault (main.py, RE2+ where the
# disk metric column is `<service>_diskio`).
FAULT_ANSWER_METRIC = {
    "cpu": "cpu",
    "mem": "mem",
    "delay": "latency",
    "loss": "latency",
    "disk": "diskio",
    "socket": "socket",
}

# Fault -> evaluation bucket (main.py's six physical classes).
FAULT_CLASS = {
    "cpu": "cpu",
    "mem": "mem",
    "disk": "io",
    "socket": "socket",
    "delay": "delay",
    "loss": "loss",
}


def answer_node(service: str, fault: str) -> Node:
    """Fine-grained ground-truth node for an injected (service, fault)."""
    return Node(service, FAULT_ANSWER_METRIC[fault])


# ---------------------------------------------------------------------------
# Candidate counting (main.py `n_candidates_map`)
# ---------------------------------------------------------------------------

def keep_non_constant(columns: Sequence[str], rows: Sequence[Sequence[float]]) -> List[str]:
    """Port of RCAEval/io/time_series.py::drop_constant.

    `df.loc[:, (df != df.iloc[0]).any()]` — a column survives when any row
    differs from the first row. NaN-vs-value comparisons count as different,
    matching pandas semantics.
    """
    if not rows:
        return list(columns)
    first = rows[0]
    kept = []
    for index, column in enumerate(columns):
        reference = first[index]
        for row in rows:
            value = row[index]
            if value != reference:
                kept.append(column)
                break
    return kept


def count_candidates(columns: Sequence[str], rows: Sequence[Sequence[float]]) -> dict:
    """RCAEval candidate counts over the windowed frame, post-preprocess.

    Mirrors main.py: `preprocess(data, dk_select_useful=False)` = drop_time +
    drop_constant (convert_mem_mb does not change columns). `rows` must align
    positionally with `columns`; time columns are stripped from both. Returns
    {"service": distinct first tokens with `-db` stripped, "metric": columns}.
    """
    time_positions = {i for i, c in enumerate(columns) if c in ("time", "Time", "time.1")}
    kept_columns = [c for i, c in enumerate(columns) if i not in time_positions]
    kept_rows = [[v for i, v in enumerate(row) if i not in time_positions] for row in rows]
    kept = keep_non_constant(kept_columns, kept_rows)
    return {
        "service": len({c.split("_")[0].replace("-db", "") for c in kept if c != "time"}),
        "metric": len([c for c in kept if c != "time"]),
    }


def candidate_universes(frame, ranking, export_strings):
    """Identities, not just denominators; constant canonical columns remain
    available because P2 ingests them. Legacy raw preprocessing is separate."""
    import hashlib
    import json
    columns = list(frame.columns)
    rows = frame.to_numpy().tolist()
    kept = keep_non_constant(columns, rows)
    raw = sorted(c for c in kept if c not in ("time", "Time", "time.1"))
    services = sorted({c.id.removeprefix("service:") for c in ranking.candidates})
    canonical = sorted({x for x in export_strings if "_" in x})
    def universe(policy, identities):
        return {"policy": policy, "identities": identities, "count": len(identities),
                "sha256": hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest()}
    return {"legacy_raw": universe("upstream_drop_time_constant_raw_v1", raw),
            "service": universe("valid_p2_services_v2", services),
            "metric": universe("canonical_available_indicator_families_v2", canonical)}


# ---------------------------------------------------------------------------
# Evaluator (RCAEval/benchmark/evaluation.py)
# ---------------------------------------------------------------------------

class Evaluator:
    """Accumulates per-case rankings and computes RCAEval's metrics.

    `max_k` is the deepest cutoff tracked; `n_candidates` per case powers the
    chance floor. Chance/lift are None when any case lacks a candidate count
    (never guessed).
    """

    def __init__(self, max_k: int = 5):
        if max_k < 1:
            raise ValueError(f"max_k must be >= 1 (got {max_k})")
        self._max_k = max_k
        self._accuracy = {k: 0.0 for k in range(1, max_k + 1)}
        self._accuracy_service = {k: 0.0 for k in range(1, max_k + 1)}
        self._anywhere = 0
        self._anywhere_service = 0
        self._rank_sizes: List[int] = []
        self._n_candidates: List[int | None] = []
        self._answer_in_candidates: List[bool] = []
        self._answer_service_position: List[int | None] = []

    def add_case(self, ranks: Sequence[Node], answer: Node,
                 n_candidates: int | None = None,
                 answer_in_candidates: bool = True) -> None:
        """Record one case; ranks is the method's ordered candidate list."""
        self._rank_sizes.append(len(ranks))
        self._n_candidates.append(n_candidates)
        self._answer_in_candidates.append(bool(answer_in_candidates))

        service_ranks = [n.entity for n in ranks]
        service_answer = answer.entity

        for k in range(1, self._max_k + 1):
            self._accuracy[k] += int(answer in ranks[:k])
            self._accuracy_service[k] += int(service_answer in service_ranks[:k])

        self._anywhere += int(answer in ranks)
        self._anywhere_service += int(service_answer in service_ranks)

        position = (service_ranks.index(service_answer) + 1
                    if service_answer in service_ranks else None)
        self._answer_service_position.append(position)

    @property
    def num(self) -> int:
        return len(self._rank_sizes)

    @property
    def n_candidates(self) -> List[int | None]:
        return list(self._n_candidates)

    @property
    def n_missing_candidates(self) -> int:
        return sum(1 for n in self._n_candidates if n is None)

    def accuracy(self, k: int) -> float | None:
        """Fine-grained AC@k."""
        if k not in self._accuracy or not self._rank_sizes:
            return None
        return self._accuracy[k] / self.num

    def accuracy_service(self, k: int) -> float | None:
        """Service-level AC@k (entity-only matching)."""
        if k not in self._accuracy_service or not self._rank_sizes:
            return None
        return self._accuracy_service[k] / self.num

    def average(self, k: int) -> float | None:
        """Avg@k = mean of AC@1..AC@k (fine-grained)."""
        if k not in self._accuracy or not self._rank_sizes:
            return None
        return sum(self.accuracy(i) for i in range(1, k + 1)) / k

    def average_service(self, k: int) -> float | None:
        """Avg@k = mean of AC@1..AC@k (service-level)."""
        if k not in self._accuracy_service or not self._rank_sizes:
            return None
        return sum(self.accuracy_service(i) for i in range(1, k + 1)) / k

    def chance_accuracy(self, k: int) -> float | None:
        """Mean over cases of min(k, n)/n for a uniform random ranking."""
        if k not in self._accuracy or not self._rank_sizes:
            return None
        if any(n is None for n in self._n_candidates):
            return None
        return sum(
            min(k, n) / n if n and found else 0.0
            for n, found in zip(self._n_candidates, self._answer_in_candidates)
        ) / self.num

    def chance_average(self, k: int) -> float | None:
        if k not in self._accuracy or not self._rank_sizes:
            return None
        parts = [self.chance_accuracy(i) for i in range(1, k + 1)]
        if any(p is None for p in parts):
            return None
        return sum(parts) / k

    def lift(self, k: int) -> float | None:
        """Avg@k minus the chance Avg@k."""
        if k not in self._accuracy or not self._rank_sizes:
            return None
        chance = self.chance_average(k)
        if chance is None:
            return None
        return self.average(k) - chance

    def mrr_service(self) -> float | None:
        """Service-level MRR (our addition; not in RCAEval's own report)."""
        if not self._rank_sizes:
            return None
        total = 0.0
        for position in self._answer_service_position:
            if position is not None:
                total += 1.0 / position
        return total / self.num

    def retrieval(self, budget: int | None = None) -> float | None:
        """Fraction of cases whose (fine-grained) answer is within budget."""
        if budget is None:
            return self._anywhere / self.num if self._rank_sizes else None
        return self.accuracy(budget)

    def retrieval_service(self, budget: int | None = None) -> float | None:
        if budget is None:
            return self._anywhere_service / self.num if self._rank_sizes else None
        return self.accuracy_service(budget)

    def rerank(self, k: int, budget: int | None = None) -> float | None:
        return self._rerank(k, budget, self.accuracy, self.retrieval)

    def rerank_service(self, k: int, budget: int | None = None) -> float | None:
        return self._rerank(k, budget, self.accuracy_service, self.retrieval_service)

    @staticmethod
    def _rerank(k, budget, accuracy, retrieval):
        if budget is not None and k > budget:
            raise ValueError(f"k must be <= budget (got k={k}, budget={budget})")
        retrieved = retrieval(budget)
        top = accuracy(k)
        if not retrieved or top is None:
            return None
        return top / retrieved
