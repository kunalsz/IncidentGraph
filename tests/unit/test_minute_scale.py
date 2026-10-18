"""Per-minute log/trace severity uses the shared baseline-scale policy.

The retired sentinel capped every log/trace z at 6 while metric z reached into
the hundreds, so severity-only ranking (`max(sev_m, sev_l, sev_tc)`) could never
let a log/trace-driven fault lead. Measured spread is now uncapped; an
unmeasured baseline is floored and persistence-guarded like the metric path.
"""

import pandas as pd
import pytest

from backend.app.eval import baseline_scale as bscale
from backend.app.eval import rcaeval_adapter as adapter


def score(pre, post, counts=True):
    return adapter._minute_score(pre, post,
                                 absolute_floor=adapter.COUNT_FLOOR if counts else bscale.ABS_FLOOR)


class TestMeasuredSpread:
    def test_large_excursion_is_not_capped(self):
        pre = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 12.0]
        out = score(pre, [10.0, 40.0])
        assert out["source"] == "measured" and out["persistent"] is True
        assert out["z"] > 6.0  # the retired sentinel ceiling
        raw = (40.0 - sum(pre) / len(pre)) / pd.Series(pre).std(ddof=1)
        assert out["z"] == pytest.approx(raw) and out["z"] < bscale.SCALE_Z_CEILING

    def test_numeric_ceiling_still_bounds_absurd_ratios(self):
        pre = [1.0, 1.0, 1.0000001]
        out = score(pre, [1e9])
        assert out["z"] == pytest.approx(bscale.SCALE_Z_CEILING)

    def test_negative_departure_keeps_sign(self):
        out = score([10.0, 10.0, 11.0], [0.0, 0.0])
        assert out["source"] == "measured" and out["z"] < 0

    def test_measured_path_ignores_persistence(self):
        # A single large minute is still a measured deviation, as for metrics.
        out = score([10.0, 10.0, 10.0, 11.0], [400.0, 10.0, 10.0, 10.0])
        assert out["persistent"] is True and out["z"] > 6.0


class TestUnmeasuredBaseline:
    def test_flat_baseline_single_extra_count_is_not_an_anomaly(self):
        out = score([0.0] * 10, [0.0, 1.0, 0.0, 0.0])
        assert (out["source"], out["persistent"], out["z"]) == ("floor", False, 0.0)

    def test_flat_baseline_persistent_burst_reaches_floor_ceiling(self):
        out = score([0.0] * 10, [0.0, 4.0, 4.0, 4.0, 0.0])
        assert out["source"] == "floor" and out["persistent"] is True
        assert out["z"] == pytest.approx(bscale.FLOOR_Z_CEILING)

    def test_two_count_minutes_qualify_but_stay_below_large_measured_values(self):
        out = score([0.0] * 10, [2.0, 2.0, 2.0])
        assert out["z"] == pytest.approx(6.0)  # 2 counts / (1 count per band/3)
        assert out["z"] < bscale.FLOOR_Z_CEILING

    def test_flat_baseline_goes_silent_is_a_persistent_drop(self):
        out = score([5.0] * 10, [0.0, 0.0, 0.0])
        assert out["source"] == "floor" and out["z"] == pytest.approx(-bscale.FLOOR_Z_CEILING)

    def test_constant_nonzero_baseline_scales_from_its_own_level(self):
        # 5% scale, so the unremarkable band is 15% of the observed level: a
        # 5% wobble is not evidence, a sustained 20% rise is.
        wobble = score([100.0] * 10, [100.0, 104.0, 105.0, 104.0])
        assert wobble["source"] == "floor" and wobble["floor_scale"] == pytest.approx(5.0)
        assert wobble["z"] == 0.0
        rise = score([100.0] * 10, [100.0, 120.0, 120.0, 120.0])
        assert rise["z"] == pytest.approx(4.0)  # 20% on a 5% scale

    def test_latency_floor_is_relative_not_a_unit_count(self):
        # count floor would make a 1-us wobble an anomaly; the latency path
        # scales from 5% of the observed p95 level instead
        pre = [50000.0, 50000.0, 50000.0]
        out = score(pre, [50000.0, 50200.0, 50200.0, 50200.0], counts=False)
        assert out["source"] == "floor"
        assert out["floor_scale"] == pytest.approx(2500.0)
        assert out["z"] < 3.0  # a 0.4% wobble is not evidence
        big = score(pre, [60000.0, 60000.0, 60000.0], counts=False)
        assert big["z"] == pytest.approx(4.0)  # 20% rise sits 4 scales out


class TestMinuteStatsIntegration:
    def test_log_error_burst_reaches_severity_above_old_cap(self):
        # pre: ten zero-count minutes; post: four error lines in three minutes
        events = pd.DataFrame({"ts": [700, 701, 702, 703, 760, 761, 762, 763,
                                      820, 821, 822, 823], "service_id": ["a"] * 12})
        scores, diag = adapter.minute_z(events, 0, 1799, 600, return_diagnostics=True)
        assert scores["a"] == pytest.approx(bscale.FLOOR_Z_CEILING)
        assert scores["a"] > 6.0  # the retired sentinel could not exceed this
        assert diag["services"]["a"]["score_source"] == "floor"

    def test_diagnostics_expose_scoring_provenance(self):
        frame = pd.DataFrame({"ts": list(range(0, 600, 30)) + [700, 760],
                              "service_id": ["a"] * 22})
        _, diag = adapter.minute_z(frame, 0, 1199, 600, return_diagnostics=True)
        info = diag["services"]["a"]
        assert info["score_source"] in ("measured", "floor")
        assert isinstance(info["score_persistent"], bool)
        assert info["peak_post"] >= 0

    def test_insufficient_baseline_is_unchanged(self):
        # Fewer than two pre-injection minutes in the window: still no score.
        frame = pd.DataFrame({"ts": [700], "service_id": ["a"]})
        scores, diag = adapter.minute_z(frame, 560, 1199, 600, return_diagnostics=True)
        assert "a" not in scores
        assert diag["services"]["a"]["status"] == "insufficient_statistics"

    def test_latency_path_uses_relative_floor(self):
        spans = pd.DataFrame({"ts": [10, 70, 700, 760, 820],
                              "service_id": ["a"] * 5, "duration_us": [50_000] * 2 + [80_000] * 3})
        scores, diag = adapter.latency_z(spans, 0, 1199, 600, return_diagnostics=True)
        info = diag["services"]["a"]
        assert info["score_source"] == "floor"
        assert info["floor_scale"] == pytest.approx(2_500.0)
        assert scores["a"] == pytest.approx(bscale.FLOOR_Z_CEILING)
