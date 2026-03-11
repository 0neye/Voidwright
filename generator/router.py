"""Backend registry for the generator module."""

from __future__ import annotations

from generator.backends.markov.backend import MarkovGeneratorBackend

__all__ = ["get_generator_backends", "get_generator_backend"]


def get_generator_backends() -> dict[str, object]:
    """Return the available generator backend instances."""

    markov_backend = MarkovGeneratorBackend()
    return {
        markov_backend.name: markov_backend,
    }


def get_generator_backend(backend_name: str):
    """Resolve one named generator backend."""

    backends = get_generator_backends()
    try:
        return backends[backend_name]
    except KeyError as exc:
        available_backends = ", ".join(sorted(backends))
        raise KeyError(
            f"Unknown generator backend '{backend_name}'. Available backends: {available_backends}"
        ) from exc
