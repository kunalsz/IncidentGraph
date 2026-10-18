"""Downstream onset retention stays bounded and does not promote summaries."""

import json
import math
import random
from types import SimpleNamespace

import pandas as pd
import pytest

from backend.app.agent.bundle import assemble_bundle
from backend.app.agent.llm import estimate_tokens
from backend.app.eval.rcaeval_adapter import series_stats
from backend.app.graph import queries
from config import tuning


ONSET = 1_000_000
SUMMARY_FIELDS = {
    'onset_clock_available', 'onset_valid_count', 'onset_qualified_count',
    'onset_longest_same_direction_run', 'onset_qualified_directions',
    'onset_representative_ts', 'onset_representative_direction',
    'post_deviation_directions',
}


def sample(sid, ts, z, *, column='opaque_cpu', **kwargs):
    return dict(id=sid, ts=ts, z=z, gamma=10.0, column=column,
                metric='cpu', direction='up' if z is None or z >= 0 else 'down',
                **kwargs)


def build(rows, **kwargs):
    return assemble_bundle('service:x', rows, [], [], token_cap=100_000, **kwargs)[0]


def ids(bundle):
    return [m['id'] for m in bundle['metrics']]


def test_all_adapter_summary_fields_are_allowlisted_and_bounded_in_every_projection():
    result = series_stats(pd.Series([-1., 0., 1.]), pd.Series([0., 4., 9., -0.5]),
                          [1000, 1012, 1410, 1500], onset_s=1000)
    assert set(result['summary']) == SUMMARY_FIELDS
    assert SUMMARY_FIELDS <= set(queries.EVIDENCE_FIELDS)
    for query in (queries.SCOPE_OBSERVATIONS, queries.SCOPE_CHANGES,
                  queries.projection('e', queries.EVIDENCE_FIELDS)):
        for field in SUMMARY_FIELDS:
            assert f'size(toJSON(e.{field})) > $field_chars' in query or (
                f'size(toJSON(c.{field})) > $field_chars' in query)
            assert f'{field}: CASE WHEN' in query
    rows = [dict(sample('onset', ONSET + 12_000, 4), **result['summary'])]
    assert build(rows, onset_ts=ONSET)['metrics'][0]['post_deviation_directions'] == 'down,up'


def crowded_rows():
    # Equal metric names must not collapse distinct exact columns.
    return [row for col in ('a_cpu', 'b_cpu', 'c_cpu') for row in (
        sample(f'{col}:peak1', ONSET + 410_000, 99, column=col),
        sample(f'{col}:peak2', ONSET + 420_000, 98, column=col),
        sample(f'{col}:late', ONSET + 20_000, 5, column=col),
        sample(f'{col}:early', ONSET + 12_000, 4, column=col),
        sample(f'{col}:context', ONSET, 0, column=col),
    )]


def test_earliest_qualified_per_exact_column_precedes_repeated_peaks_and_is_deterministic():
    rows = crowded_rows()
    expected = [f'{col}:early' for col in ('a_cpu', 'b_cpu', 'c_cpu')]
    assert ids(build(rows, onset_ts=ONSET, max_metrics=3)) == expected
    full = build(rows, onset_ts=ONSET, max_metrics=15)
    assert ids(full)[:3] == expected
    assert ids(full)[3:6] == [f'{col}:peak1' for col in ('a_cpu', 'b_cpu', 'c_cpu')]
    assert ids(full)[6:9] == [f'{col}:peak2' for col in ('a_cpu', 'b_cpu', 'c_cpu')]
    assert {m['id'] for m in full['metrics'] if m['z'] == 0} == {
        f'{col}:context' for col in ('a_cpu', 'b_cpu', 'c_cpu')}
    random.Random(7).shuffle(rows)
    assert build(rows, onset_ts=ONSET, max_metrics=15) == full


def test_token_cap_drops_extras_before_any_primary_representative():
    rows = crowded_rows()
    primary = build(rows, onset_ts=ONSET, max_metrics=3)
    cap = estimate_tokens(json.dumps(primary))
    bundle, _ = assemble_bundle('service:x', rows, [], [], onset_ts=ONSET,
                                max_metrics=15, token_cap=cap)
    assert ids(bundle) == ids(primary)
    assert estimate_tokens(json.dumps(bundle)) <= cap
    one = build(rows, onset_ts=ONSET, max_metrics=1)
    # Input totals remain unchanged, so this is the exact one-row envelope cap.
    capped, _ = assemble_bundle('service:x', rows, [], [], onset_ts=ONSET,
                                token_cap=estimate_tokens(json.dumps(one)))
    assert ids(capped) == ['a_cpu:early']


@pytest.mark.parametrize('clock', [None, math.nan, math.inf, True])
def test_without_authoritative_onset_selection_is_exact_legacy(clock):
    rows = crowded_rows()
    expected = sorted(rows, key=lambda m: (-abs(m.get('z') or 0), m['id']))[:7]
    assert build(rows, onset_ts=clock, max_metrics=7)['metrics'] == expected


@pytest.mark.parametrize('patch', [
    {'z': None, 'gamma': None},  # no baseline
    {'z': 2.99},
    {'gamma': None},  # modality alone cannot pass SVAR
    {'gamma': 2.99},
    {'gamma': math.inf},
    {'direction': 'unknown'},
    {'direction': 'down'},  # sign mismatch
    {'ts': ONSET + tuning.ONSET_DELTA_MS + 1},
    {'ts': None},
])
def test_summary_cannot_rescue_unqualified_rows(patch):
    rejected = sample('false-onset', ONSET + 1_000, 4)
    rejected.update(onset_clock_available=True, onset_qualified_count=99,
                    onset_representative_ts=ONSET + 1_000,
                    onset_representative_direction='up',
                    onset_qualified_directions='up', post_deviation_directions='up')
    rejected.update(patch)
    peak = sample('peak', ONSET + 410_000, 99)
    assert ids(build([rejected, peak], onset_ts=ONSET, max_metrics=1)) == ['peak']


@pytest.mark.parametrize('sign', [-1, 1])
def test_inclusive_threshold_boundary_and_window_filter(sign):
    boundary = sample('boundary', ONSET + tuning.ONSET_DELTA_MS,
                      sign * tuning.NSIGMA_N)
    boundary['gamma'] = tuning.NSIGMA_N
    peak = sample('peak', ONSET + 410_000, sign * 99)
    assert ids(build([peak, boundary], onset_ts=ONSET, max_metrics=1)) == ['boundary']
    window = SimpleNamespace(t0=ONSET, t1=boundary['ts'] - 1)
    assert ids(build([peak, boundary], onset_ts=ONSET, window=window, max_metrics=1)) == ['peak']


def test_no_qualifying_onset_still_fairly_reserves_each_series_peak():
    rows = [sample('a:1', ONSET + 410_000, 99, column='a'),
            sample('a:2', ONSET + 420_000, 98, column='a'),
            sample('b:1', ONSET + 410_000, 4, column='b')]
    assert ids(build(rows, onset_ts=ONSET, max_metrics=2)) == ['a:1', 'b:1']
    assert ids(build(rows, max_metrics=2)) == ['a:1', 'a:2']


def test_summaries_do_not_change_gate_or_make_causal_claims():
    row = sample('onset', ONSET + 12_000, 4)
    row.update(post_deviation_directions='down,up', onset_qualified_directions='down,up',
               onset_qualified_count=1, onset_longest_same_direction_run=1)
    peak = sample('peak', ONSET + 410_000, 99)
    kept = build([peak, row], onset_ts=ONSET, max_metrics=1)['metrics'][0]
    assert kept == row
    assert 'claim_eligible' not in kept
    assert 'causal' not in kept


def test_metric_limit_zero_remains_empty():
    assert build(crowded_rows(), onset_ts=ONSET, max_metrics=0)['metrics'] == []
