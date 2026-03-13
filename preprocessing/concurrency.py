"""Shared concurrency helpers for preprocessing stages."""

from __future__ import annotations

import argparse
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
import os
from typing import Any, Callable, Iterable, Literal, TypeVar

__all__ = [
    "ExecutorMode",
    "WorkResult",
    "add_concurrency_arguments",
    "resolve_executor_mode",
    "resolve_worker_count",
    "create_executor_factory",
    "run_auto_parallel_work",
]

ExecutorMode = Literal["auto", "thread", "process"]
WorkResult = TypeVar("WorkResult")

_AUTO_STAGE_EXECUTORS: dict[str, Literal["thread", "process"]] = {
    "extract": "thread",
    "canonicalize": "process",
    "canonicalize_write": "thread",
    "graphs": "process",
    "graph_expansion": "process",
}


def add_concurrency_arguments(
    parser: argparse.ArgumentParser,
    *,
    worker_flag: str = "--workers",
    executor_flag: str = "--executor",
    help_prefix: str,
) -> None:
    """Add common concurrency CLI flags to a parser.

    Args:
        parser: Parser that should receive the concurrency flags
        worker_flag: CLI flag name for worker-count overrides
        executor_flag: CLI flag name for executor selection
        help_prefix: Human-readable stage prefix used in help text
    """

    parser.add_argument(
        worker_flag,
        type=int,
        default=None,
        help=(
            f"Worker count for {help_prefix}. Defaults to an auto-sized value "
            "based on the executor type and available hardware."
        ),
    )
    parser.add_argument(
        executor_flag,
        choices=("auto", "thread", "process"),
        default="auto",
        help=(
            f"Executor type for {help_prefix}. `auto` picks the stage default "
            "for the current workload."
        ),
    )


def resolve_executor_mode(stage_name: str, requested_mode: ExecutorMode) -> Literal["thread", "process"]:
    """Resolve the effective executor mode for a preprocessing stage.

    Args:
        stage_name: Known preprocessing stage name
        requested_mode: CLI or API executor selection

    Returns:
        The concrete executor type that should be used
    """

    if requested_mode != "auto":
        return requested_mode
    if stage_name not in _AUTO_STAGE_EXECUTORS:
        raise ValueError(
            f"Unknown preprocessing stage: {stage_name!r}. "
            "Register it in _AUTO_STAGE_EXECUTORS."
        )
    return _AUTO_STAGE_EXECUTORS[stage_name]


def resolve_worker_count(
    *,
    task_count: int,
    stage_name: str,
    requested_workers: int | None,
    requested_mode: ExecutorMode,
) -> int:
    """Return a hardware-aware worker count for one stage.

    Args:
        task_count: Number of independent tasks in the stage
        stage_name: Known preprocessing stage name
        requested_workers: Optional explicit worker-count override
        requested_mode: Requested executor selection

    Returns:
        A positive worker count capped to the available task count
    """

    capped_task_count = max(1, task_count)
    if requested_workers is not None:
        return max(1, min(requested_workers, capped_task_count))

    effective_mode = resolve_executor_mode(stage_name, requested_mode)
    cpu_count = max(1, os.cpu_count() or 1)

    if effective_mode == "thread":
        auto_workers = min(32, cpu_count * 4)
    else:
        auto_workers = cpu_count if cpu_count <= 2 else cpu_count - 1

    return max(1, min(auto_workers, capped_task_count))


def create_executor_factory(executor_mode: Literal["thread", "process"]) -> type[Executor]:
    """Return the concrete executor class for an executor mode.

    Args:
        executor_mode: Concrete executor mode chosen for the stage

    Returns:
        The matching standard-library executor class
    """

    return ThreadPoolExecutor if executor_mode == "thread" else ProcessPoolExecutor


def run_auto_parallel_work(
    *,
    stage_name: str,
    requested_mode: ExecutorMode,
    worker_count: int,
    submit_work: Callable[[type[Executor]], Iterable[WorkResult]],
) -> tuple[list[WorkResult], Literal["thread", "process"]]:
    """Run stage work with automatic process-to-thread fallback.

    Args:
        stage_name: Known preprocessing stage name
        requested_mode: User-requested executor mode
        worker_count: Concrete worker count for this stage
        submit_work: Callback that receives an executor class and returns results

    Returns:
        The collected results and the concrete executor mode that succeeded

    Raises:
        Any executor or task error when fallback is not allowed or when the
        fallback attempt also fails
    """

    primary_mode = resolve_executor_mode(stage_name, requested_mode)
    primary_executor_factory = create_executor_factory(primary_mode)
    try:
        return list(submit_work(primary_executor_factory)), primary_mode
    except (NotImplementedError, OSError, PermissionError):
        if requested_mode != "auto" or primary_mode != "process":
            raise

    fallback_mode: Literal["thread", "process"] = "thread"
    fallback_executor_factory = create_executor_factory(fallback_mode)
    return list(submit_work(fallback_executor_factory)), fallback_mode
