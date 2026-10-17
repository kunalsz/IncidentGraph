"""In-memory retained phase state, not a claim of durable graph-only resume."""
from dataclasses import dataclass, field
from typing import Any
import uuid

@dataclass
class RunState:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: str = "running"
    failed_stage: str | None = None
    errors: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    investigation: Any = None
    scope: Any = None
    ranking: Any = None
    invest: Any = None
    validated: Any = None
    fused: Any = None
    report: Any = None
    budget: dict = field(default_factory=dict)
    runtime: dict = field(default_factory=dict)

    def model_dump(self):
        return {name: value.model_dump() if hasattr(value, "model_dump") else value
                for name, value in vars(self).items()}

    def model_dump_json(self, **kwargs):
        import json
        return json.dumps(self.model_dump(), **kwargs)
