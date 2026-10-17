"""Bounded template reads. LIMIT bounds rows, never scanning or sorting work."""
import hashlib
import json
from dataclasses import dataclass, field
from backend.app.settings import SCOPE_SETTINGS


class PayloadAdmissionError(ValueError):
    pass


def admit_payload(value, *, field_chars=4096, max_items=128, depth=0):
    """Reject, never silently trim, nested/unbounded or non-finite payloads."""
    import math
    if depth > 4:
        raise PayloadAdmissionError('payload nesting exceeds admission limit')
    if value is None or type(value) is bool:
        return
    if type(value) in (int, float):
        if not math.isfinite(value):
            raise PayloadAdmissionError('non-finite payload')
        return
    if isinstance(value, str):
        if len(value) > field_chars:
            raise PayloadAdmissionError('field exceeds admission limit')
        return
    if isinstance(value, (list, tuple, dict)):
        if len(value) > max_items:
            raise PayloadAdmissionError('collection exceeds admission limit')
        for key, item in value.items() if isinstance(value, dict) else enumerate(value):
            if isinstance(key, str) and len(key) > field_chars:
                raise PayloadAdmissionError('key exceeds admission limit')
            admit_payload(item, field_chars=field_chars, max_items=max_items, depth=depth+1)
        return
    raise PayloadAdmissionError('unsupported payload type')


def _digest(rows):
    # Node compatibility is only for legacy SDK/test adapters; production templates
    # below return maps, not nodes or unbounded property bags.
    def default(value):
        if hasattr(value, 'properties'):
            return {'properties': value.properties, 'labels': value.labels}
        raise TypeError(type(value).__name__)
    return hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False,
                                     default=default).encode()).hexdigest()


@dataclass
class ReadGateway:
    graph: object
    settings: object = SCOPE_SETTINGS
    diagnostics: list = field(default_factory=list)
    transcript: list = field(default_factory=list)

    def read(self, query, params=None, *, cap):
        if type(cap) is not int or cap < 1:
            raise ValueError('read cap must be positive')
        if len(self.diagnostics) >= self.settings.max_queries:
            raise RuntimeError('graph query budget exhausted')
        params = dict(params or {})
        params.update(read_limit=cap + 1, field_chars=getattr(self.settings, 'field_chars', 4096))
        query = query.rstrip().rstrip(';') + ' LIMIT $read_limit'
        record = {'cap': cap, 'timeout_ms': self.settings.query_timeout_ms,
                  'overflow': False, 'status': 'pending'}
        self.diagnostics.append(record)
        try:
            rows = self.graph.query(query, params=params,
                                    timeout=self.settings.query_timeout_ms).result_set
            digest = _digest(rows)
        except Exception as exc:
            record.update(status='error', error_type=type(exc).__name__)
            raise
        self.transcript.append((query, params, digest))
        record.update(status='ok', overflow=len(rows) > cap, returned=min(cap, len(rows)))
        return rows[:cap], len(rows) > cap

    def validate(self):
        """Optimistic repeated-read check, NOT a transactional database snapshot."""
        transcript = list(self.transcript)
        for query, params, expected in transcript:
            if len(self.diagnostics) >= self.settings.max_queries:
                raise RuntimeError('graph query budget exhausted during snapshot validation')
            record = {'status': 'validating', 'timeout_ms': self.settings.query_timeout_ms}
            self.diagnostics.append(record)
            rows = self.graph.query(query, params=params,
                                    timeout=self.settings.query_timeout_ms).result_set
            if _digest(rows) != expected:
                record['status'] = 'source_changed'
                raise ValueError('scope source changed; rerun P1')
            record['status'] = 'validated'
        # Recheck generation after the scan as well. Publication uses a fresh
        # marker for each successful import; raw writes still rely on row replay.
        if transcript and '__incidentgraph_import_publication__' in transcript[0][1].values():
            query, params, expected = transcript[0]
            if len(self.diagnostics) >= self.settings.max_queries:
                raise RuntimeError('graph query budget exhausted during generation validation')
            rows = self.graph.query(query, params=params, timeout=self.settings.query_timeout_ms).result_set
            self.diagnostics.append({'status': 'generation_validated', 'timeout_ms': self.settings.query_timeout_ms})
            if _digest(rows) != expected:
                raise ValueError('source publication changed; rerun P1')
        return transcript
