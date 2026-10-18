"""Sparse observations cannot establish a persistent baseline departure."""

import math

import pandas as pd
import pytest

from backend.app.engine import policy
from backend.app.eval import baseline_scale as bs
from backend.app.eval import rcaeval_adapter as adapter


def series(before, after):
    return adapter.series_stats(pd.Series(before, dtype=float), pd.Series(after, dtype=float),
                                list(range(100, 100 + len(after))))


@pytest.mark.parametrize("post", [[50.0], [50.0, 50.0],
                                   [math.nan] * 599 + [50.0],
                                   [50.0, math.nan, 50.0]])
def test_floor_requires_three_observed_departures(post):
    scale = bs.baseline_scale([0.0] * 20)
    assert not bs.persistent(post, scale)
    assert bs.gamma(post, scale) == 0.0
    out = series([0.0] * 20, post)
    assert out["stats"]["gamma"] == 0.0
    assert all(sample["z"] == 0.0 for sample in out["samples"])


def test_three_observed_departures_still_qualify_with_missing_samples():
    out = series([0.0] * 20, [50.0, math.nan, 50.0, math.nan, 50.0])
    assert out["stats"]["gamma"] == 10.0
    assert out["samples"][0]["z"] == 10.0


@pytest.mark.parametrize("before", [[], [math.nan] * 20,
                                    [math.inf, -math.inf, math.nan]])
def test_missing_baseline_is_unavailable_not_constant_zero(before):
    scale = bs.baseline_scale(before)
    assert scale.source == "unavailable"
    assert not bs.persistent([50.0] * 20, scale)
    assert bs.gamma([50.0] * 20, scale) is None
    assert bs.deviation(50.0, scale) is None
    out = series(before, [50.0] * 20)
    assert out["stats"]["scale_source"] == "unavailable"
    assert out["stats"]["gamma"] is None
    for name in ("baseline_mean", "baseline_std", "baseline_median", "baseline_iqr"):
        assert out["stats"][name] is None
    assert out["samples"]
    for sample in out["samples"]:
        assert sample["z"] is None
        assert not sample["z_capped"]
        assert policy.modality("MetricAnomaly", {**sample, "gamma": None}) is None


def test_no_finite_post_observations_emit_no_samples():
    out = series([0.0] * 20, [math.nan, math.inf, -math.inf])
    assert out["samples"] == []


def test_missing_baseline_and_missing_post_remain_unscored():
    out = series([], [math.nan])
    assert out["samples"] == []
    assert out["stats"]["gamma"] is None
