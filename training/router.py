"""Backend registry for the training module."""

from __future__ import annotations

from training.backends.markov.backend import MarkovTrainingBackend
from training.backends.hgt.backend import HGTTrainingBackend

__all__ = ["get_training_backends", "get_training_backend"]


def get_training_backends() -> dict[str, object]:
    """Return the available training backend instances."""

    markov_backend = MarkovTrainingBackend()
    hgt_backend = HGTTrainingBackend()
    return {
        markov_backend.name: markov_backend,
        hgt_backend.name: hgt_backend,
    }


def get_training_backend(backend_name: str):
    """Resolve one named training backend."""

    backends = get_training_backends()
    try:
        return backends[backend_name]
    except KeyError as exc:
        available_backends = ", ".join(sorted(backends))
        raise KeyError(
            f"Unknown training backend '{backend_name}'. Available backends: {available_backends}"
        ) from exc
