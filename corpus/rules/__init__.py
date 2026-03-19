"""Built-in corpus filter rules registry."""

from __future__ import annotations

from corpus.rules.base import CorpusRule

__all__ = ["CorpusRule", "BUILTIN_RULES"]

# Populated lazily to avoid circular imports; importers should call
# build_active_ruleset() from corpus.filter instead of this list directly.
BUILTIN_RULES: list[type[CorpusRule]] = []
