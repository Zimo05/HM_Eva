from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JobSpec:
    dataset: str
    model: str
    condition: str = "full"
    kind: str = "stationary"
    script: str = ""

    @property
    def job_key(self) -> str:
        return ":".join((self.kind, self.dataset, self.model, self.condition))
