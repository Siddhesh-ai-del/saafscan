"""Immutable verdict model — the single contract shared by pipeline, API and UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ADULTERANTS = frozenset({"none", "starch", "detergent", "both", "inconclusive"})
SEVERITIES = frozenset({"none", "low", "moderate", "high"})


@dataclass(frozen=True)
class Verdict:
    adulterant: str
    confidence: float
    severity: str
    advice: str
    evidence: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.adulterant not in ADULTERANTS:
            raise ValueError(f"unknown adulterant: {self.adulterant!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence out of range: {self.confidence}")
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity: {self.severity!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "adulterant": self.adulterant,
            "confidence": round(self.confidence, 4),
            "severity": self.severity,
            "advice": self.advice,
            "evidence": {k: round(float(v), 2) for k, v in self.evidence.items()},
        }
