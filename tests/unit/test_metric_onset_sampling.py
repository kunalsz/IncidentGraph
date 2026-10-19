"""Bounded, label-independent metric sampling; no database or model required."""

import math

import pandas as pd
import pytest

from backend.app.eval import rcaeval_adapter as adapter
from config import tuning


ONSET = 1000
BASE = [-1.0, 0.0, 1.0]


def series(after, *, before=BASE, times=None, onset_s=ONSET):
    return adapter.series_stats(
        pd.Series(before, dtype=float), pd.Series(after, dtype=float),
        times if times is not None else list(range(ONSET, ONSET + len(after))),
        onset_s=onset_s,
    )


@pytest.mark.parametrize("firstqual,peak", [(12, 410), (20, 478)])
@pytest.mark.parametrize("sign", [1, -1])
def test_delayed_peak_keeps_early_qualifying_onset(firstqual, peak, sign):
    post = [0.0] * 600
    post[firstqual] = sign * 4.0
    post[peak] = sign * 9.0
    out = series(post)
    legacy = series(post, onset_s=None)
    assert out["stats"] == legacy["stats"]
    assert out["samples"][:2] == legacy["samples"]
    assert [s["ts"] for s in out["samples"]] == [
        (ONSET + peak) * 1000, ONSET * 1000, (ONSET + firstqual) * 1000]
    assert out["samples"][2]["z"] == sign * 4.0
    summary = out["summary"]
    assert summary["onset_qualified_count"] == 1
    assert summary["onset_longest_same_direction_run"] == 1
    assert summary["onset_qualified_directions"] == ("up" if sign > 0 else "down")
    assert summary["onset_representative_ts"] == (ONSET + firstqual) * 1000


@pytest.mark.parametrize("sign", [1, -1])
def test_threshold_and_onset_boundary_are_inclusive(sign):
    assert tuning.ONSET_DELTA_MS % 1000 == 0
    boundary = ONSET + tuning.ONSET_DELTA_MS // 1000
    out = series([0.0, sign * tuning.NSIGMA_N, sign * 9.0],
                 times=[ONSET, boundary, boundary + 1])
    assert out["summary"]["onset_valid_count"] == 2
    assert out["summary"]["onset_qualified_count"] == 1
    assert out["summary"]["onset_representative_ts"] == boundary * 1000
    assert len(out["samples"]) == 3


def test_gamma_gate_is_required_even_if_z_qualifies(monkeypatch):
    # Measured std and IQR differ: this value crosses only the z gate.
    monkeypatch.setattr(tuning, "NSIGMA_N", 3.0)
    out = series([0.0, 5.0, 6.0], before=[-2.0, -1.0, 0.0, 1.0, 2.0])
    assert abs(out["samples"][0]["z"]) > 3
    assert out["stats"]["gamma"] == 3  # exact gamma threshold also qualifies
    assert out["summary"]["onset_representative_ts"] == (ONSET + 1) * 1000
    rejected = series([0.0, 4.9, 5.9], before=[-2.0, -1.0, 0.0, 1.0, 2.0])
    assert abs(rejected["samples"][0]["z"]) > 3
    assert rejected["stats"]["gamma"] < 3
    assert rejected["summary"]["onset_qualified_count"] == 0
    assert len(rejected["samples"]) == 2


@pytest.mark.parametrize("post,times", [
    ([0.0, 2.99, 2.0], [ONSET, ONSET + 1, ONSET + 2]),
        ([0.0, 4.0, 9.0], [ONSET, ONSET + tuning.ONSET_DELTA_MS // 1000 + 1, ONSET + 410]),
])
def test_no_qualifying_onset_keeps_only_legacy(post, times):
    out = series(post, times=times)
    assert out["samples"] == series(post, times=times, onset_s=None)["samples"]
    assert out["summary"]["onset_qualified_count"] == 0
    assert out["summary"]["onset_representative_ts"] is None
    assert out["summary"]["onset_representative_direction"] == ""
    assert out["summary"]["onset_qualified_directions"] == ""


@pytest.mark.parametrize("before", [[], [math.nan], [math.inf, -math.inf]])
def test_missing_baseline_retains_unscored_legacy_values(before):
    out = series([0.0, 5.0, 9.0], before=before)
    assert out["stats"]["gamma"] is None
    assert all(s["z"] is None for s in out["samples"])
    assert len(out["samples"]) == 2
    assert out["summary"]["onset_valid_count"] == 3
    assert out["summary"]["onset_qualified_count"] == 0
    assert out["summary"]["post_deviation_directions"] == ""


@pytest.mark.parametrize("post", [[0.0, 50.0], [0.0, 50.0, math.nan, 50.0]])
def test_sparse_floor_does_not_change_global_persistence_policy(post):
    out = series(post, before=[0.0] * 20)
    assert out["stats"]["gamma"] == 0
    assert all(s["z"] == 0 for s in out["samples"])
    assert out["summary"]["onset_qualified_count"] == 0


def test_global_floor_can_qualify_isolated_onset_without_implying_persistence():
    out = series([0.0, 50.0, math.nan, 50.0, 50.0], before=[0.0] * 20,
                 times=[ONSET, ONSET + 12, ONSET + 13, ONSET + 410, ONSET + 411])
    assert out["stats"]["gamma"] == 10
    assert out["samples"][0]["z"] == 10
    assert out["summary"]["onset_qualified_count"] == 1
    assert out["summary"]["onset_longest_same_direction_run"] == 1


def test_runs_break_on_missing_nonqualifying_and_opposite_direction_rows():
    out = series([0, 4, 5, math.nan, 4, 0, -4, -5, -6, 4, math.inf, 4])
    assert out["summary"]["onset_valid_count"] == 10
    assert out["summary"]["onset_qualified_count"] == 8
    assert out["summary"]["onset_longest_same_direction_run"] == 3
    assert out["summary"]["onset_qualified_directions"] == "down,up"
    assert out["summary"]["onset_representative_direction"] == "up"


def test_full_post_sign_reversal_disclosed_even_when_unsampled_and_subthreshold():
    out = series([0, 4, 9, -0.5], times=[ONSET, ONSET + 12, ONSET + 410, ONSET + 500])
    assert all(s["direction"] == "up" for s in out["samples"])
    assert out["summary"]["onset_qualified_directions"] == "up"
    assert out["summary"]["post_deviation_directions"] == "down,up"


@pytest.mark.parametrize("post,times,expected", [
    ([4, 9], [ONSET, ONSET + 1], 2),  # onset is first
    ([0, 9], [ONSET, ONSET + 1], 2),  # onset is peak
    ([9], [ONSET], 1),
    ([0, 4, 9], [ONSET, ONSET + 1, ONSET + 1], 2),  # peak wins timestamp tie
    ([0, 4, 9], [ONSET, ONSET, ONSET], 1),
])
def test_samples_have_at_most_three_unique_timestamps(post, times, expected):
    out = series(post, times=times)
    assert len(out["samples"]) == expected
    assert len({s["ts"] for s in out["samples"]}) == expected
    assert out["samples"][0]["value"] == max(post)


def test_many_qualifying_rows_still_emit_only_three():
    out = series([0.0] + [4.0] * 59 + [9.0])
    assert len(out["samples"]) == 3
    assert out["summary"]["onset_qualified_count"] == 60
    assert out["summary"]["onset_longest_same_direction_run"] == 60


def test_no_authoritative_onset_has_explicit_no_fallback():
    late = ONSET + tuning.ONSET_DELTA_MS // 1000 + 100
    post, times = [0.0, 4.0, 9.0], [late, late + 12, late + 410]
    out = adapter.series_stats(pd.Series(BASE), pd.Series(post), times)
    assert len(out["samples"]) == 2
    assert out["summary"]["onset_clock_available"] is False
    assert out["summary"]["onset_valid_count"] == 0
    assert out["summary"]["onset_representative_ts"] is None
    assert series(post, times=times)["summary"]["onset_valid_count"] == 0


@pytest.mark.parametrize("post", [[], [math.nan, math.inf, -math.inf]])
def test_empty_finite_post_has_explicit_empty_summary(post):
    out = series(post)
    assert out["samples"] == []
    assert out["summary"]["onset_clock_available"] is True
    assert out["summary"]["onset_valid_count"] == 0
    assert out["summary"]["onset_qualified_count"] == 0
    assert out["summary"]["onset_representative_ts"] is None


def test_earliest_means_timestamp_not_first_source_position():
    out = series([0.0, 4.0, 5.0, 9.0],
                 times=[ONSET, ONSET + 20, ONSET + 12, ONSET + 410])
    assert out["summary"]["onset_representative_ts"] == (ONSET + 12) * 1000
    assert out["samples"][-1]["value"] == 5.0


def test_load_case_uses_injection_clock_and_propagates_scalar_summary(monkeypatch, tmp_path):
    # Anonymous column; onset is NOT the first post observation timestamp.
    metrics = pd.DataFrame({
        "time": [997, 998, 999, 1005, 1012, 1410],
        "anonymous_latency-90": BASE + [0.0, 4.0, 9.0],
    })
    monkeypatch.setattr(adapter, "read_case", lambda _: {
        "inject_s": ONSET, "metrics": metrics, "logs": None, "traces": None})
    original = adapter.series_stats
    clocks = []

    def capture(before, after, times, onset_s=None):
        clocks.append(onset_s)
        return original(before, after, times, onset_s=onset_s)

    monkeypatch.setattr(adapter, "series_stats", capture)

    class Graph:
        def __init__(self):
            self.metric_rows = []

        def query(self, query, params):
            if "MERGE (n:MetricAnomaly" in query:
                self.metric_rows.extend(params["rows"])

    graph = Graph()
    # Anonymous synthetic case: state the topology instead of relying on the
    # per-suite inference the runner uses for real case directories.
    result = adapter.load_case(graph, tmp_path / "opaque", topology_path=adapter.OB_TOPOLOGY)
    assert clocks == [ONSET]
    assert result["counts"]["metrics"] == len(graph.metric_rows) == 3
    expected = series([0, 4, 9], times=[1005, 1012, 1410])["summary"]
    for row in graph.metric_rows:
        assert {name: row[name] for name in expected} == expected
        assert all(value is None or isinstance(value, (str, int, float, bool))
                   for value in expected.values())
    pd.testing.assert_frame_equal(metrics, pd.DataFrame({
        "time": [997, 998, 999, 1005, 1012, 1410],
        "anonymous_latency-90": BASE + [0.0, 4.0, 9.0],
    }))
