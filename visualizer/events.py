"""Shared generation-visualization event records and recorder helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

__all__ = [
    "VisualizationEvent",
    "VisualizationEventSink",
    "VisualizationPart",
    "VisualizationRecorder",
]


@dataclass(frozen=True)
class VisualizationPart:
    """One concrete world-space part placement used for visualization."""

    part_id: str
    rotation: int
    x: int
    y: int
    flip_x: bool = False
    flip_y: bool = False


@dataclass(frozen=True)
class VisualizationEvent:
    """One recorded visualization event for a generated sample."""

    kind: str
    sample_index: int
    message: str = ""
    part: VisualizationPart | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class VisualizationEventSink(Protocol):
    """Behavior required by generation backends that emit visualization events."""

    def sample_started(
        self,
        *,
        config: Mapping[str, Any] | None = None,
        seeded: bool = False,
    ) -> None:
        """Record the start of one generated sample."""

    def part_placed(
        self,
        *,
        part: VisualizationPart,
        message: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Record one accepted placement."""

    def attempt_rejected(
        self,
        *,
        reason: str,
        part: VisualizationPart | None = None,
        message: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Record one rejected generation attempt."""

    def sample_finished(
        self,
        *,
        stats: Mapping[str, Any],
        stop_reason: str,
        message: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Record the completion of one generated sample."""


class VisualizationRecorder:
    """In-memory recorder for one generated sample's visualization events."""

    def __init__(self, sample_index: int):
        self.sample_index = sample_index
        self.events: list[VisualizationEvent] = []

    def _append(
        self,
        kind: str,
        *,
        message: str = "",
        part: VisualizationPart | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.events.append(
            VisualizationEvent(
                kind=kind,
                sample_index=self.sample_index,
                message=message,
                part=part,
                metadata=dict(metadata or {}),
            )
        )

    def sample_started(
        self,
        *,
        config: Mapping[str, Any] | None = None,
        seeded: bool = False,
    ) -> None:
        self._append(
            "sample_started",
            message=f"sample-{self.sample_index:03d} started",
            metadata={"config": dict(config or {}), "seeded": seeded},
        )

    def part_placed(
        self,
        *,
        part: VisualizationPart,
        message: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._append("part_placed", message=message, part=part, metadata=metadata)

    def attempt_rejected(
        self,
        *,
        reason: str,
        part: VisualizationPart | None = None,
        message: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        event_metadata = dict(metadata or {})
        event_metadata.setdefault("reason", reason)
        self._append("attempt_rejected", message=message, part=part, metadata=event_metadata)

    def sample_finished(
        self,
        *,
        stats: Mapping[str, Any],
        stop_reason: str,
        message: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        event_metadata = dict(metadata or {})
        event_metadata.setdefault("stats", dict(stats))
        event_metadata.setdefault("stop_reason", stop_reason)
        self._append("sample_finished", message=message, metadata=event_metadata)
