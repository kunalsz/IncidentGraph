"""Unit tests for the RCAEval adapter (windowing + series stats). No graph needed."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from backend.app.eval import baseline_scale as bscale  # noqa: E402
from backend.app.eval import rcaeval_adapter as adapter  # noqa: E402
from backend.app.eval import rcaeval_runner as runner  # noqa: E402
from fetch_rcaeval import parse_case, select_subset  # noqa: E402


def synthetic_metrics(n_before=700, n_after=741):
    times = list(range(1000, 1000 + n_before + n_after))
    inject = times[n_before]
    frame = pd.DataFrame({
        "time": times,
        "a_cpu": [1.0] * (n_before + n_after),
        "a_latency-90": [0.01] * (n_before + n_after),
        "a_latency-50": [0.01] * (n_before + n_after),
    })
    return frame, inject


class TestWindowFrames:
    def test_slices_match_rcaeval_protocol(self):
        metrics, inject = synthetic_metrics()
        slices = adapter.window_frames(metrics, inject)
        # last 600 before, first 600 from inject (main.py semantics)
        assert len(slices["before"]) == 600
        assert len(slices["after"]) == 600
        assert slices["before"]["time"].iloc[-1] == inject - 1
        assert slices["after"]["time"].iloc[0] == inject
        assert len(slices["frame"]) == 1200
        assert slices["t0_ms"] == (inject - 600) * 1000
        assert slices["t1_ms"] == (inject + 599) * 1000

    def test_indices_are_positional(self):
        metrics, inject = synthetic_metrics()
        slices = adapter.window_frames(metrics, inject)
        assert list(slices["after"].index[:3]) == [0, 1, 2]
        assert list(slices["before"].index[:3]) == [0, 1, 2]


class TestMetricColumns:
    def test_excludes_time_and_latency_50_maps_latency_90(self):
        metrics, _ = synthetic_metrics()
        columns = adapter.metric_columns(metrics)
        names = [c for c, _, _ in columns]
        assert "time" not in names
        assert "a_latency-50" not in names
        assert ("a_latency-90", "a", "latency") in columns
        assert ("a_cpu", "a", "cpu") in columns

    def test_entity_split_on_first_underscore(self):
        frame = pd.DataFrame({"time": [0], "frontend-external_error": [1.0]})
        columns = adapter.metric_columns(frame)
        assert columns == [("frontend-external_error", "frontend-external", "error")]


class TestSeriesStats:
    def _series(self, before_values, after_values):
        before = pd.Series(before_values)
        after = pd.Series(after_values)
        times = list(range(100, 100 + len(after_values)))
        return adapter.series_stats(before, after, times)

    def test_peak_and_onset_samples(self):
        stats = self._series([0.0, 1.0, 0.0, 1.0], [0.0, 5.0, 0.0, 0.0])
        samples = stats["samples"]
        assert len(samples) == 2
        peak = max(samples, key=lambda s: abs(s["z"]))
        assert peak["value"] == 5.0
        assert peak["direction"] == "up"
        # onset sample is the first post-injection row
        onset = min(samples, key=lambda s: s["ts"])
        assert onset["ts"] == 100 * 1000

    def test_zero_variance_baseline_scores_persistent_change(self):
        # Flat baseline, window that leaves it and stays out: scored by the
        # floor-scale policy (flagged), no longer the Z_CAP sentinel. The full
        # policy is covered in test_baseline_scale.py.
        stats = self._series([2.0] * 20, [2.0] + [7.0] * 20)
        peak = max(stats["samples"], key=lambda s: abs(s["z"]))
        assert abs(peak["z"]) >= 3
        assert peak["z_capped"] is True
        assert stats["stats"]["gamma"] >= 3

    def test_zero_variance_single_blip_is_not_an_anomaly(self):
        stats = self._series([2.0] * 20, [2.0, 7.0, 2.0, 2.0])
        assert all(s["z"] == 0.0 for s in stats["samples"])
        assert stats["stats"]["gamma"] == 0.0

    def test_zero_variance_no_change_is_zero(self):
        stats = self._series([2.0, 2.0], [2.0, 2.0])
        assert all(sample["z"] == 0.0 for sample in stats["samples"])
        assert all(not sample["z_capped"] for sample in stats["samples"])

    def test_nan_window_rows_skipped(self):
        stats = self._series([0.0, 1.0, 0.0], [float("nan"), 2.0])
        assert len(stats["samples"]) >= 1
        assert all(sample["z"] == sample["z"] for sample in stats["samples"])

    def test_gamma_and_baseline_stats(self):
        stats = self._series([1.0, 1.0, 1.0, 1.0, 3.0], [8.0])
        # median 1.0, IQR 0 -> stored IQR stays 0, but gamma now comes from the
        # next scale on the ladder (std) instead of silently becoming 0.0.
        assert stats["stats"]["baseline_median"] == 1.0
        assert stats["stats"]["baseline_iqr"] == 0.0
        assert stats["stats"]["scale_source"] == "std"
        assert stats["stats"]["gamma"] > 3


class TestMinuteZ:
    def test_post_activity_vs_zero_baseline(self):
        events = pd.DataFrame({"ts": [10, 700], "service_id": ["a", "a"]})
        # Complete grids include nine zero baseline bins.
        z = adapter.minute_z(events, 0, 1199, 600)
        assert z["a"] == pytest.approx(2.8460498941515415)
        # No pre events, one post minute: a single quiet observation against an
        # unmeasured baseline is not an anomaly (was the flat Z_CAP sentinel).
        events2 = pd.DataFrame({"ts": [700], "service_id": ["a"]})
        scores, diag = adapter.minute_z(events2, 0, 1199, 600, return_diagnostics=True)
        assert scores["a"] == 0.0
        assert diag["services"]["a"]["score_source"] == "floor"
        assert diag["services"]["a"]["score_persistent"] is False
        # The same baseline with a persistent multi-count burst scores at the
        # floor ceiling: decisive, but below a large measured deviation.
        events3 = pd.DataFrame({"ts": [700, 701, 702, 703, 760, 761, 762, 763,
                                       820, 821, 822, 823], "service_id": ["a"] * 12})
        assert adapter.minute_z(events3, 0, 1199, 600)["a"] == pytest.approx(
            bscale.FLOOR_Z_CEILING)


class TestRunnerHelpers:
    def test_parse_case_name(self):
        assert runner.parse_case_name("re2ob_checkoutservice_cpu_1") == (
            "checkoutservice", "cpu", "1")
        assert runner.parse_case_name("re2ob_productcatalogservice_loss_3") == (
            "productcatalogservice", "loss", "3")

    def test_export_ranks_merges_service_order_with_indicators(self):
        from backend.app.engine.contracts import RankedCandidate, RankResult

        def candidate(service, rank):
            return RankedCandidate(id=f"service:{service}", rank=rank, score=1.0 - rank / 10,
                                   components={}, component_ranks={}, evidence=[],
                                   rule_flags=[], selected=True)

        result = RankResult(
            incident_id="i", candidates=[candidate("checkoutservice", 1),
                                         candidate("currencyservice", 2),
                                         candidate("redis", 3)],
            branch_ids=["service:checkoutservice"], indicators=[
                {"service_id": "service:checkoutservice", "indicator": "cpu",
                 "gamma": 240.0, "evidence": []},
                {"service_id": "service:checkoutservice", "indicator": "mem",
                 "gamma": 5.0, "evidence": []},
                {"service_id": "service:currencyservice", "indicator": "latency",
                 "gamma": 9.0, "evidence": []},
            ], audit=[], n_candidates=3, truncated=False, config={})
        ranks = runner.export_ranks(result)
        # service order from P2; indicators gamma-desc inside each block;
        # redis has no indicators -> bare name
        assert ranks == ["checkoutservice_cpu", "checkoutservice_mem",
                         "currencyservice_latency", "redis"]

    def test_export_ranks_uses_fused_order_in_agent_mode(self):
        from backend.app.engine.contracts import (FuseResult, FusedHypothesis,
                                                  RankedCandidate, RankResult)

        def candidate(service, rank):
            return RankedCandidate(id=f"service:{service}", rank=rank, score=1.0 - rank / 10,
                                   components={}, component_ranks={}, evidence=[],
                                   rule_flags=[], selected=True)

        ranking = RankResult(
            incident_id="i", candidates=[candidate("checkoutservice", 1),
                                         candidate("currencyservice", 2),
                                         candidate("redis", 3)],
            branch_ids=[], indicators=[
                {"service_id": "service:checkoutservice", "indicator": "cpu",
                 "gamma": 10.0, "evidence": []},
                {"service_id": "service:currencyservice", "indicator": "latency",
                 "gamma": 9.0, "evidence": []},
            ], audit=[], n_candidates=3, truncated=False, config={})

        def fused_item(hid, service, h):
            return FusedHypothesis(id=hid, candidate=f"service:{service}", rank=1,
                                   h_score=h, components={}, confidence=0.8)

        fused = FuseResult(
            incident_id="i", presentation="multi",
            fused=[fused_item("h1", "currencyservice", 0.9),
                   fused_item("h2", "redis", 0.5)],
            winner="h1", winner_candidate="service:currencyservice")
        ranks = runner.export_ranks(ranking, fused=fused)
        # Default remains P2 even if proposals exist; explicit diagnostic opt-in.
        assert ranks[0].startswith("checkoutservice")
        assert runner.export_ranks(ranking, fused=fused, policy="eligible_fusion")[0].startswith("currencyservice")
        fused.fused[0].status = "disproved"
        assert not runner.export_ranks(ranking, fused=fused, policy="eligible_fusion")[0].startswith("currencyservice")


class TestFetchSelection:
    def test_subset_is_deterministic_and_spread(self):
        cases = [f"re2ob_{s}_{f}_{i}"
                 for s in ["adservice", "cartservice", "checkoutservice",
                           "currencyservice", "emailservice"]
                 for f in ["cpu", "delay", "disk", "loss", "mem", "socket"]
                 for i in (1, 2, 3)]
        picked_a = select_subset(cases, 20, seed=0)
        picked_b = select_subset(cases, 20, seed=0)
        assert picked_a == picked_b
        assert len(picked_a) == 20
        faults = {parse_case(case)[1] for case in picked_a}
        services = {parse_case(case)[0] for case in picked_a}
        assert faults == {"cpu", "delay", "disk", "loss", "mem", "socket"}
        assert len(services) == 5

    def test_subset_covers_all_when_requested_larger(self):
        cases = [f"re2ob_svc_cpu_{i}" for i in (1, 2, 3)]
        assert select_subset(cases, 99, seed=0) == sorted(cases)
