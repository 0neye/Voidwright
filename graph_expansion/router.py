"""Backend registry for the graph expansion module."""

from __future__ import annotations

from graph_expansion.backends.structural.backend import StructuralExpansionBackend

__all__ = ["get_expansion_backends", "get_expansion_backend"]


def get_expansion_backends() -> dict[str, object]:
    """Return the available expansion backend instances."""

    structural_backend = StructuralExpansionBackend()
    return {structural_backend.name: structural_backend}


def get_expansion_backend(backend_name: str):
    """Resolve one named expansion backend."""

    backends = get_expansion_backends()
    try:
        return backends[backend_name]
    except KeyError as exc:
        available_backends = ", ".join(sorted(backends))
        raise KeyError(
            f"Unknown expansion backend '{backend_name}'. Available backends: {available_backends}"
        ) from exc
