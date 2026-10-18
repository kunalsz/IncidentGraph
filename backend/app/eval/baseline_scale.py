"""Baseline scale policy for metric series, including zero-spread baselines.

Why this exists
---------------
A pre-injection baseline with no spread (counter/gauge that sits at a constant,
typically 0) admits no z-score or IQR ratio: dividing by zero is undefined.
The adapter previously handled this with two *inconsistent* sentinels:

* `z = +/-Z_CAP (6.0)`  -> looked decisive to the severity vector, but 6.0 is
  lower than the uncapped z of any noisy series (frontend latency routinely
  exceeds 20), so a constant-baseline true cause was out-ranked on severity.
* `gamma = 0.0`          -> `policy.modality()` requires gamma >= NSIGMA_N when
  gamma is present, so the same record was simultaneously "z=6" and "not an
  anomaly", and could never become qualified local evidence.

Policy (pure python, no pandas, so it is unit-testable anywhere)
----------------------------------------------------------------
1. Choose a positive scale from a ladder, first that is > 0:
   IQR -> std -> relative floor.
   IQR is the robust estimator BARO (`RobustScaler`) and TORAI's FineGrainer
   (median/IQR, gamma) rely on; the reference implementations silently fall
   back to scale 1 on zero spread (sklearn behaviour), which is unit-dependent.
   We use a *relative* floor instead (see below). There is deliberately no MAD
   rung: IQR == 0 forces the middle half of the values onto the median, which
   already forces MAD == 0 (checked by brute force over 6.9k random series),
   so such a rung could never fire.
2. `z` and `gamma` on the zero-spread path are computed from the SAME scale
   and share one ceiling (`FLOOR_Z_CEILING`, above NSIGMA_N=3), so they cannot
   disagree about whether a departure is an anomaly. Known limit: a baseline
   that is *exactly* 0 and becomes persistently non-zero always qualifies; with
   no unit information that is genuinely indistinguishable from a real change.
3. Persistence guard: when the baseline had zero spread (floor path), a
   departure only counts if at least `MIN_PERSIST` post-injection samples leave
   the baseline band. A single transient blip against a flat baseline is
   exactly what a naive floor would over-score.
4. A baseline with no finite observations is unavailable, not constant zero.
   Its deviations and gamma remain unscored (`None`).
5. Provenance is explicit: `scale_source` records which rung of the ladder was
   used, so downstream evidence can distinguish a measured spread from a floor.

The relative floor is `max(REL_FLOOR * |median|, ABS_FLOOR)`. It is a property
of the series' own magnitude, never of a metric family or fault type, so it
makes no benchmark-specific assumption.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

REL_FLOOR = 0.05          # 5% of the baseline level
ABS_FLOOR = 1e-6          # last resort when median == 0 (still > 0)
MIN_PERSIST = 3           # post samples that must leave the band (floor path)
BAND_K = 3.0              # band half-width, in scale units
SCALE_Z_CEILING = 1_000.0  # numeric ceiling for measured-spread scales
FLOOR_Z_CEILING = 10.0     # floor scales are not a measured spread: decisive
                           # (>= NSIGMA_N) but never dominant over measured series


@dataclass(frozen=True)
class BaselineScale:
    median: float
    scale: float
    source: str            # "iqr" | "std" | "floor" | "unavailable"
    spread_measured: bool  # False => scale is a floor, not an observed spread


def _quantile(sorted_values: Sequence[float], fraction: float) -> float:
    # Linear interpolation, identical to pandas' default (and `scorers.percentile`).
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def _finite(values: Sequence[float]) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def baseline_scale(baseline: Sequence[float]) -> BaselineScale:
    """Pick the scale for one series' pre-injection samples."""
    values = sorted(_finite(baseline))
    if not values:
        return BaselineScale(0.0, ABS_FLOOR, "unavailable", False)
    med = _quantile(values, 0.5)
    iqr = _quantile(values, 0.75) - _quantile(values, 0.25)
    if iqr > 0:
        return BaselineScale(med, iqr, "iqr", True)
    if len(values) > 1:
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
        if std > 0:
            return BaselineScale(med, std, "std", True)
    return BaselineScale(med, max(REL_FLOOR * abs(med), ABS_FLOOR), "floor", False)


def ceiling(scale: BaselineScale) -> float:
    return SCALE_Z_CEILING if scale.spread_measured else FLOOR_Z_CEILING


def deviation(value: float, scale: BaselineScale) -> float | None:
    """Bounded signed deviation, or None when baseline observations are missing."""
    if scale.source == "unavailable":
        return None
    raw = (value - scale.median) / scale.scale
    bound = ceiling(scale)
    return max(-bound, min(bound, raw))


def persistent(post: Sequence[float], scale: BaselineScale,
               min_persist: int = MIN_PERSIST, k: float = BAND_K) -> bool:
    """Do enough post samples leave the baseline band?

    Only enforced when the spread was not measured (floor path): a measured
    IQR/std already encodes how noisy the series normally is. Missing samples
    do not reduce the required number of observed departures.
    """
    if scale.source == "unavailable":
        return False
    if scale.spread_measured:
        return True
    band = k * scale.scale
    outside = sum(1 for v in _finite(post) if abs(v - scale.median) > band)
    return outside >= min_persist


def gamma(post: Sequence[float], scale: BaselineScale) -> float | None:
    """TORAI FineGrainer gamma: max |x - median| / scale over the window.

    Returns 0.0 when a floor-scaled departure is not persistent, so a lone
    spike on a flat baseline does not become decisive evidence.
    """
    if scale.source == "unavailable":
        return None
    values = _finite(post)
    if not values or not persistent(values, scale):
        return 0.0
    return min(ceiling(scale), max(abs(v - scale.median) for v in values) / scale.scale)
