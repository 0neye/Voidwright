"""Token history serialization helpers for Markov states."""

from __future__ import annotations

from typing import Sequence

from .types import END_TOKEN, RelativePlacementToken

__all__ = ["history_symbol", "state_key"]


def history_symbol(token_key: str) -> str:
    """Return the compact history symbol used for state-key serialization"""

    if token_key == END_TOKEN:
        return END_TOKEN
    token = RelativePlacementToken.from_key(token_key)
    return f"{token.part_id}|{token.rotation}"


def state_key(history: Sequence[str], order: int) -> str:
    """Return the serialized Markov state key for a history tail

    Args:
        history: Full token-key history in emission order
        order: Number of tail elements to include

    Returns:
        Compact state key that can index transition count maps
    """

    tail = list(history[-order:]) if order > 0 else []
    compact_history = [history_symbol(token_key) for token_key in tail]
    return " || ".join(compact_history)
