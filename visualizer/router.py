"""Registry for static ship visualization backends."""

from __future__ import annotations

from visualizer.backends.base import StaticVisualizationBackend
from visualizer.backends.cardinal_zones import CardinalZonesBackend
from visualizer.backends.spatial_zones import SpatialZonesBackend
from visualizer.backends.traversable_clusters import TraversableClustersBackend

__all__ = ["get_static_backend", "get_static_backends"]

_BACKENDS: dict[str, StaticVisualizationBackend] = {
    backend.name: backend
    for backend in (
        SpatialZonesBackend(),
        CardinalZonesBackend(),
        TraversableClustersBackend(),
    )
}


def get_static_backends() -> dict[str, StaticVisualizationBackend]:
    """Return all registered static visualization backends keyed by name."""
    return dict(_BACKENDS)


def get_static_backend(name: str) -> StaticVisualizationBackend:
    """Return the backend registered under *name*.

    Raises
    ------
    KeyError
        If no backend is registered with that name.
    """
    backend = _BACKENDS.get(name)
    if backend is None:
        available = ", ".join(sorted(_BACKENDS))
        raise KeyError(f"Unknown visualization backend {name!r}. Available: {available}")
    return backend
