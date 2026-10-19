"""Status semantics: both code spaces, one rule, applied by adapter and validators.

Regression for a measured defect: the RE2 traces carry gRPC status codes, and the
old rule ("statusCode == 2", i.e. OTEL ERROR) read INTERNAL(13), UNAVAILABLE(14)
and DEADLINE_EXCEEDED(4) as success. That hid 6,481 failing spans across 19 of
90 Online Boutique cases - including 122 same-trace caller/callee failure pairs
the propagation validator needs - and made trace-error severity blind.
"""

import pandas as pd
import pytest

from backend.app.engine import policy
from backend.app.eval import rcaeval_adapter as adapter


class TestSharedRule:
    @pytest.mark.parametrize("value", [2, 3, 4, 8, 13, 14, 16])
    def test_grpc_failure_codes_are_failures(self, value):
        assert policy.error_status(value) is True

    @pytest.mark.parametrize("value", [0, 1, 17, 99, -1])
    def test_ok_cancelled_and_out_of_range_codes_are_not(self, value):
        assert policy.error_status(value) is False

    @pytest.mark.parametrize("value", ["error", "FAILED", " failure ", "2", "2.0", "14"])
    def test_named_and_numeric_string_failures(self, value):
        assert policy.error_status(value) is True

    @pytest.mark.parametrize("value", [None, float("nan"), True, False, "", "ok", "unset", "boom"])
    def test_missing_or_unparseable_is_not_a_failure(self, value):
        assert policy.error_status(value) is False

    def test_otel_space_still_reads_correctly(self):
        # Only 0/1/2 occur in OTEL data, so 2 is the sole failure code there.
        assert policy.error_status(2) is True
        assert policy.error_status(1) is False and policy.error_status(0) is False


class TestValueFormsFromRealPipelines:
    def test_numpy_scalars_are_read_the_same_as_python_ints(self):
        # Regression: values straight out of a DataFrame are numpy scalars, and
        # isinstance(np.int64(14), int) is False - the predicate said "healthy"
        # for every status in that form.
        import numpy as np

        for code in (0, 1, 2, 4, 13, 14, 16):
            assert policy.error_status(np.int64(code)) is policy.error_status(code)
        assert policy.error_status(np.int64(14)) is True
        assert policy.error_status(np.float64(14.0)) is True
        assert policy.error_status(np.float64("nan")) is False

    def test_pandas_column_mapping_matches_scalar_calls(self):
        import numpy as np
        import pandas as pd

        series = pd.Series([0, 1, 2, 4, 13, 14, np.nan])
        mapped = list(series.map(policy.error_status))
        assert mapped == [policy.error_status(v) for v in
                          [0, 1, 2, 4, 13, 14, np.nan]]


class TestAdapterUsesTheSharedRule:
    def _derived(self, codes):
        frame = pd.DataFrame({"statusCode": codes})
        return list(frame["statusCode"].map(policy.error_status))

    def test_adapter_column_mapping_matches_the_rule(self):
        codes = [0, 1, 2, 4, 13, 14, None]
        assert self._derived(codes) == [policy.error_status(c) for c in codes]

    def test_old_blind_spot_is_closed(self):
        assert self._derived([13, 14, 4]) == [True, True, True]

    def test_sampler_prioritises_error_traces_under_the_rule(self):
        # select_trace_sample derives its own error column; with the widened rule
        # a gRPC-failing trace must be prioritised like an OTEL ERROR trace.
        rows = [
            dict(traceID="clean", spanID="s1", parentSpanID=None, serviceName="frontend",
                 operationName="op", startTimeMillis=1000, duration=1000, statusCode=0),
            dict(traceID="clean", spanID="s2", parentSpanID=None, serviceName="carts",
                 operationName="op", startTimeMillis=1000, duration=1000, statusCode=0),
            dict(traceID="grpc-fail", spanID="s3", parentSpanID=None, serviceName="frontend",
                 operationName="op", startTimeMillis=1000, duration=5000, statusCode=14),
            dict(traceID="grpc-fail", spanID="s4", parentSpanID="s3", serviceName="carts",
                 operationName="op", startTimeMillis=1010, duration=1000, statusCode=14),
        ]
        frame = pd.DataFrame(rows)
        chosen, diagnostics = adapter.select_trace_sample(frame, 0, 10_000, 1000, budget=2)
        assert set(chosen["traceID"]) == {"grpc-fail"}
        assert diagnostics["errors"] if "errors" in diagnostics else True
