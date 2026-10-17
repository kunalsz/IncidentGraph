"""Validated, immutable development read limits; not production tuning claims."""
from dataclasses import dataclass, asdict
import hashlib
import json


@dataclass(frozen=True)
class ScopeSettings:
    raw_observations: int = 2000
    changes: int = 200
    changes_per_service: int = 5
    trace_context: int = 500
    query_timeout_ms: int = 2000
    max_queries: int = 512
    field_chars: int = 4096
    change_context_ms: int = 86_400_000
    fuzzy_margin: float = 0.10
    version: str = "scope-v3-projected-repeat-read"

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key not in ('version', 'fuzzy_margin'):
                if type(value) is not int or value < 1:
                    raise ValueError(f'{key} must be a positive integer')
        if not 0 <= self.fuzzy_margin <= 1:
            raise ValueError('fuzzy_margin must be within [0,1]')

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class LinkingSettings:
    """Independent P0 catalog limits; does not change frozen scope profiles."""
    max_catalog: int = 2000
    query_timeout_ms: int = 2000
    max_queries: int = 8
    field_chars: int = 4096

    def __post_init__(self):
        for key, value in asdict(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f'{key} must be a positive integer')


@dataclass(frozen=True)
class RuntimeSettings:
    db_host: str = 'localhost'
    db_port: int = 6380
    deployment_mode: str = 'local'
    parallelism: int = 8

    def __post_init__(self):
        from config import tuning
        if (not isinstance(self.db_host, str) or not self.db_host.strip()
                or any(c.isspace() for c in self.db_host) or '/' in self.db_host):
            raise ValueError('database host must be a usable host name or address')
        if type(self.db_port) is not int or not 1 <= self.db_port <= 65535:
            raise ValueError('database port must be within 1..65535')
        if self.deployment_mode != 'local':
            raise ValueError('only local deployment mode is currently supported')
        if type(self.parallelism) is not int or not 1 <= self.parallelism <= tuning.MAX_AGENT_PARALLEL:
            raise ValueError('parallelism exceeds the configured agent ceiling')

    @classmethod
    def from_environment(cls, environ=None):
        import os
        from config import tuning
        env = os.environ if environ is None else environ
        return cls(db_host=env.get('FALKORDB_HOST', 'localhost'),
                   db_port=int(env.get('FALKORDB_PORT', '6380')),
                   deployment_mode=env.get('INCIDENTGRAPH_DEPLOYMENT_MODE', 'local'),
                   parallelism=int(env.get('INCIDENTGRAPH_PARALLELISM', str(tuning.MAX_AGENT_PARALLEL))))

    @property
    def redacted_snapshot(self):
        # Credentials/provider keys are intentionally not part of this schema.
        from types import MappingProxyType
        return MappingProxyType(asdict(self))

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def validate_startup_settings(environ=None):
    return RuntimeSettings.from_environment(environ)


LINKING_SETTINGS = LinkingSettings()
SCOPE_SETTINGS = ScopeSettings()

# Explicit cold-ranking profile; callers must opt in and persist the effective hash.
SCOPE_BENCHMARK_SETTINGS = ScopeSettings(raw_observations=20000, trace_context=2000,
                                         max_queries=1024, version='scope-benchmark-v3-projected-repeat-read')
