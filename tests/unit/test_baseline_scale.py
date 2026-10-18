"""Constant-baseline scoring policy + authoritative level plumbing. No graph needed."""

import json

import pandas as pd
import pytest

from backend.app.eval import baseline_scale as bs
from backend.app.eval import rcaeval_adapter as adapter
from backend.app.engine import policy
from backend.app.graph import validators
from config import tuning


def stats(before, after):
    return adapter.series_stats(pd.Series(before), pd.Series(after), list(range(100, 100 + len(after))))


def qualifies(props):
    """The exact gate P4 applies to a cited metric record."""
    return policy.modality("MetricAnomaly", props) == "metrics"


class TestScaleLadder:
    def test_iqr_first(self):
        s = bs.baseline_scale([1, 2, 3, 4, 5, 6, 7, 8])
        assert (s.source, s.spread_measured) == ("iqr", True)

    def test_std_rung_when_iqr_is_zero_but_series_is_not_constant(self):
        # A majority of identical values forces IQR == 0; the rare excursions
        # still give a measured std.
        values = [0.0] * 90 + [1.0] * 10
        s = bs.baseline_scale(values)
        assert (s.source, s.spread_measured) == ("std", True) and s.scale > 0

    def test_iqr_zero_implies_mad_zero_so_no_mad_rung_is_needed(self):
        import random
        from statistics import median
        rng = random.Random(7)
        checked = 0
        for _ in range(3000):
            vals = sorted(float(rng.choice([0, 0, 0, 1, 2, 5])) for _ in range(rng.randint(2, 30)))
            if bs._quantile(vals, .75) - bs._quantile(vals, .25) > 0:
                continue
            med = bs._quantile(vals, .5)
            assert median(sorted(abs(v - med) for v in vals)) == 0
            checked += 1
        assert checked > 100

    def test_constant_baseline_uses_relative_floor(self):
        s = bs.baseline_scale([100.0] * 50)
        assert (s.source, s.spread_measured) == ("floor", False)
        assert s.scale == pytest.approx(5.0)  # 5% of the level, not a unit-dependent 1.0

    def test_zero_constant_uses_absolute_floor_still_positive(self):
        s = bs.baseline_scale([0.0] * 50)
        assert s.source == "floor" and s.scale > 0

    def test_empty_and_nan(self):
        assert bs.baseline_scale([]).scale > 0
        assert bs.baseline_scale([float("nan")] * 3).scale > 0


class TestConstantBaselineQualifies:
    """The bug: z=6 with gamma=0 was 'not an anomaly' at the modality gate."""

    def test_constant_zero_to_persistent_change_is_qualified_evidence(self):
        out = stats([0.0] * 600, [0.0] * 100 + [50.0] * 500)
        peak = max(out["samples"], key=lambda s: abs(s["z"]))
        assert out["stats"]["gamma"] >= tuning.NSIGMA_N
        assert abs(peak["z"]) >= tuning.NSIGMA_N
        assert peak["z_capped"] is True  # floor-scaled, flagged
        assert qualifies({"z": peak["z"], "gamma": out["stats"]["gamma"]})

    def test_z_and_gamma_cannot_disagree_on_zero_spread_path(self):
        for after in ([0.0] * 100 + [5.0] * 500, [0.0] * 599 + [5.0], [0.0] * 600):
            out = stats([0.0] * 600, after)
            gamma = out["stats"]["gamma"]
            for sample in out["samples"]:
                assert (abs(sample["z"]) >= tuning.NSIGMA_N) == (gamma >= tuning.NSIGMA_N) or sample["z"] == 0.0

    def test_isolated_blip_on_flat_baseline_is_not_evidence(self):
        out = stats([0.0] * 600, [0.0] * 300 + [50.0] + [0.0] * 299)
        assert out["stats"]["gamma"] == 0.0
        assert all(sample["z"] == 0.0 for sample in out["samples"])

    def test_no_change_stays_zero(self):
        out = stats([7.0] * 600, [7.0] * 600)
        assert out["stats"]["gamma"] == 0.0
        assert all(s["z"] == 0.0 and not s["z_capped"] for s in out["samples"])

    def test_small_wobble_on_constant_baseline_is_not_evidence(self):
        # within the 3-scale band of the relative floor (5% of 100 -> band 15)
        out = stats([100.0] * 600, [100.0] * 300 + [103.0] * 300)
        assert out["stats"]["gamma"] < tuning.NSIGMA_N

    def test_floor_deviation_never_dominates_measured_series(self):
        out = stats([0.0] * 600, [5000.0] * 600)
        assert max(abs(s["z"]) for s in out["samples"]) <= bs.FLOOR_Z_CEILING
        assert out["stats"]["gamma"] <= bs.FLOOR_Z_CEILING


class TestMeasuredPathsUnchanged:
    def test_std_positive_z_is_plain_zscore(self):
        before = [0.0, 1.0, 0.0, 1.0]
        out = stats(before, [0.0, 5.0, 0.0, 0.0])
        s = pd.Series(before)
        peak = max(out["samples"], key=lambda x: abs(x["z"]))
        assert peak["z"] == pytest.approx((5.0 - s.mean()) / s.std(ddof=1))
        assert peak["z_capped"] is False

    def test_iqr_positive_gamma_is_median_iqr_ratio(self):
        before = list(range(100))
        out = stats(before, [500.0])
        s = pd.Series(before)
        iqr = s.quantile(.75) - s.quantile(.25)
        assert out["stats"]["gamma"] == pytest.approx(abs(500 - s.median()) / iqr)

    def test_rare_blip_baseline_gets_consistent_measured_gamma(self):
        # std > 0 but IQR == 0: previously gamma=0.0 while z was huge
        before = [0.0] * 590 + [1.0] * 10
        out = stats(before, [50.0] * 600)
        assert out["stats"]["baseline_iqr"] == 0.0
        assert out["stats"]["scale_source"] == "std"
        assert out["stats"]["gamma"] >= tuning.NSIGMA_N
        peak = max(out["samples"], key=lambda s: abs(s["z"]))
        assert qualifies({"z": peak["z"], "gamma": out["stats"]["gamma"]})

    def test_old_sentinel_pair_was_rejected_by_gate(self):
        # documents the defect this change fixes
        assert not qualifies({"z": adapter.Z_CAP, "gamma": 0.0})


TOPOLOGY = json.loads(adapter.OB_TOPOLOGY.read_text())
ENTITIES = {s["name"] for s in TOPOLOGY["services"]}


class TestAuthoritativeLevels:
    def test_levels_are_derived_not_assumed(self):
        depths = adapter.dependency_depths(TOPOLOGY, ENTITIES)
        assert depths["service:frontend"] == 1
        assert depths["service:cartservice"] == 2        # BFS, not longest path (3)
        assert depths["service:redis"] == 3
        assert "service:istio-init" not in depths         # isolated: no basis for a depth

    def test_only_derivable_services_are_marked_authoritative(self):
        auth = adapter.authoritative_levels(TOPOLOGY, ENTITIES)
        assert "service:frontend" in auth and "service:emailservice" in auth
        assert "service:istio-init" not in auth

    def test_mismatched_authored_level_is_not_authoritative(self):
        broken = json.loads(json.dumps(TOPOLOGY))
        next(s for s in broken["services"] if s["id"] == "service:cartservice")["level"] = 9
        assert "service:cartservice" not in adapter.authoritative_levels(broken, ENTITIES)

    def test_entity_restriction_option_changes_depths_so_load_case_must_not_use_it(self):
        # Restricting to a telemetry subset makes `frontend` a root (depth 0 != authored 1).
        assert adapter.dependency_depths(TOPOLOGY, {"frontend", "adservice"}) == \
            {"service:frontend": 0, "service:adservice": 1}
        assert "service:frontend" not in adapter.authoritative_levels(TOPOLOGY, {"frontend", "adservice"})
        # ...which is why the full-topology derivation is the one that is used:
        assert "service:frontend" in adapter.authoritative_levels(TOPOLOGY)

    def test_convention_matches_what_p4_accepts(self):
        # No drift between the writer (adapter) and the reader (p4_validate).
        from backend.app.engine import p4_validate
        nodes = [{"id": sid, "type": "Service",
                  "properties": {"level": TOPOLOGY_LEVEL[sid], "level_convention": adapter.LEVEL_CONVENTION}}
                 for sid in ("service:frontend", "service:currencyservice")]
        scope = type("S", (), {"nodes": nodes})()
        levels = p4_validate._service_levels(scope)
        assert levels == {"service:frontend": 1, "service:currencyservice": 2}
        verdict = validators.level_veto(levels["service:currencyservice"], levels["service:frontend"])
        assert verdict["verdict"] == "pass"  # was always "partial": authoritative level unavailable


TOPOLOGY_LEVEL = {s["id"]: s["level"] for s in TOPOLOGY["services"]}
