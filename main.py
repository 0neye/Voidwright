"""Unified root entry point for the ship generator pipeline."""

from __future__ import annotations

import argparse
import ctypes
import os
import shlex
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, TextIO

from generator import cli as generator_cli
from preprocessing import cli as preprocessing_cli
from training import cli as training_cli

__all__ = ["main"]


@dataclass(frozen=True)
class CommandDomain:
    """Describe one top-level command domain."""

    name: str
    help_text: str
    build_parser: Callable[[], argparse.ArgumentParser]
    run: Callable[[Sequence[str] | None], int]


def build_domain_registry() -> dict[str, CommandDomain]:
    """Build the top-level command registry.

    The registry intentionally delegates to the existing package CLIs so newly
    registered backends and subcommands continue to work without changes here.
    """

    # Keep the registry data-driven so future domains can be added in one place
    command_domains = (
        CommandDomain(
            name="preprocessing",
            help_text="Run preprocessing stages over local ship image corpora",
            build_parser=preprocessing_cli.build_parser,
            run=preprocessing_cli.main,
        ),
        CommandDomain(
            name="training",
            help_text="Build and validate backend-specific training artifacts",
            build_parser=training_cli.build_parser,
            run=training_cli.main,
        ),
        CommandDomain(
            name="generator",
            help_text="Generate ship outputs with the available backends",
            build_parser=generator_cli.build_parser,
            run=generator_cli.main,
        ),
    )
    return {command_domain.name: command_domain for command_domain in command_domains}


def build_root_parser(command_registry: dict[str, CommandDomain]) -> argparse.ArgumentParser:
    """Build the root parser for the unified CLI."""

    available_entrypoints = ", ".join(
        ["commands", "help", "repl", *sorted(command_registry)]
    )
    parser = argparse.ArgumentParser(
        description=(
            "Unified ship generator entry point for preprocessing, training, "
            "generation, and interactive REPL use."
        ),
        epilog=(
            "Examples:\n"
            "  python main.py preprocessing pipeline downloaded_ships --verbose\n"
            "  python main.py training build markov --graph-input-dir generated_ship_graphs_canonical\n"
            "  python main.py generator generate markov --output-dir out/generated-ships\n"
            "  python main.py repl"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "entrypoint",
        nargs="?",
        help=f"Top-level command to run ({available_entrypoints})",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the selected entry point",
    )
    return parser


def find_subparsers_action(
    parser: argparse.ArgumentParser,
) -> argparse._SubParsersAction | None:
    """Return the subparser action for a parser when present."""

    for parser_action in parser._actions:
        if isinstance(parser_action, argparse._SubParsersAction):
            return parser_action
    return None


def iter_leaf_command_paths(
    parser: argparse.ArgumentParser,
    prefix_tokens: tuple[str, ...] = (),
) -> Iterable[tuple[str, ...]]:
    """Yield leaf command paths from an argparse parser tree."""

    subparsers_action = find_subparsers_action(parser)
    if subparsers_action is None:
        yield prefix_tokens
        return

    # Walk the parser tree so command discovery stays automatic
    for command_name, child_parser in sorted(subparsers_action.choices.items()):
        yield from iter_leaf_command_paths(child_parser, prefix_tokens + (command_name,))


def print_available_commands(
    command_registry: dict[str, CommandDomain],
    output_stream: TextIO,
) -> None:
    """Print the discovered command paths."""

    output_stream.write("Available commands:\n")
    for command_domain_name in sorted(command_registry):
        command_domain = command_registry[command_domain_name]
        output_stream.write(f"  {command_domain.name}: {command_domain.help_text}\n")
        for command_tokens in iter_leaf_command_paths(
            command_domain.build_parser(),
            prefix_tokens=(command_domain.name,),
        ):
            output_stream.write(f"    {' '.join(command_tokens)}\n")


def normalize_exit_code(raw_exit_code: object) -> int:
    """Normalize command exit codes returned by delegated CLIs."""

    if raw_exit_code is None:
        return 0
    if isinstance(raw_exit_code, int):
        return raw_exit_code
    return 1


def invoke_command(
    command_runner: Callable[[Sequence[str] | None], int],
    command_arguments: Sequence[str],
) -> int:
    """Invoke a delegated command and normalize argparse exits."""

    try:
        return normalize_exit_code(command_runner(command_arguments))
    except SystemExit as system_exit:
        return normalize_exit_code(system_exit.code)


def run_help_command(
    help_arguments: Sequence[str],
    command_registry: dict[str, CommandDomain],
    root_parser: argparse.ArgumentParser,
    output_stream: TextIO,
) -> int:
    """Show help for the root CLI or a delegated command."""

    if not help_arguments:
        root_parser.print_help(file=output_stream)
        return 0

    entrypoint_name = help_arguments[0]
    command_domain = command_registry.get(entrypoint_name)
    if command_domain is None:
        output_stream.write(
            f"Unknown help target '{entrypoint_name}'. Use 'commands' to list options.\n"
        )
        return 2

    child_parser = command_domain.build_parser()
    child_parser.prog = f"{root_parser.prog} {entrypoint_name}"
    resolved_parser, unmatched_tokens = resolve_help_parser(
        child_parser,
        help_arguments[1:],
    )
    if unmatched_tokens:
        output_stream.write(
            f"Unknown help target '{' '.join(help_arguments)}'. Use 'commands' to list options.\n"
        )
        return 2

    resolved_parser.print_help(file=output_stream)
    return 0


def run_domain_command(
    entrypoint_name: str,
    command_arguments: Sequence[str],
    command_registry: dict[str, CommandDomain],
    error_stream: TextIO,
) -> int:
    """Dispatch a top-level command domain."""

    command_domain = command_registry.get(entrypoint_name)
    if command_domain is None:
        available_entrypoints = ", ".join(sorted(command_registry))
        error_stream.write(
            f"Unknown entry point '{entrypoint_name}'. Available domains: {available_entrypoints}\n"
        )
        return 2
    return invoke_command(command_domain.run, command_arguments)


def split_repl_command_line(raw_command_line: str) -> list[str]:
    """Split a REPL command line using the platform-appropriate rules."""

    if os.name == "nt":
        return split_windows_command_line(raw_command_line)

    return shlex.split(raw_command_line, posix=True)


def split_windows_command_line(raw_command_line: str) -> list[str]:
    """Split a command line with Windows parsing semantics."""

    command_line_to_argv = ctypes.windll.shell32.CommandLineToArgvW
    command_line_to_argv.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
    command_line_to_argv.restype = ctypes.POINTER(ctypes.c_wchar_p)
    local_free = ctypes.windll.kernel32.LocalFree
    local_free.argtypes = [ctypes.c_void_p]
    local_free.restype = ctypes.c_void_p

    argument_count = ctypes.c_int()
    argument_vector = command_line_to_argv(raw_command_line, ctypes.byref(argument_count))
    if not argument_vector:
        raise ValueError("Unable to parse command line")

    try:
        # CommandLineToArgvW strips wrapping quotes while preserving Windows paths
        return [argument_vector[index] for index in range(argument_count.value)]
    finally:
        local_free(argument_vector)


def resolve_help_parser(
    parser: argparse.ArgumentParser,
    help_arguments: Sequence[str],
) -> tuple[argparse.ArgumentParser, list[str]]:
    """Resolve the parser that matches a help target path.

    Args:
        parser: Parser to traverse from
        help_arguments: Nested command tokens after the domain name

    Returns:
        A tuple containing the most specific resolved parser and any unmatched
        tokens that did not correspond to a subcommand path
    """

    current_parser = parser
    remaining_tokens = list(help_arguments)

    # Walk only through subcommand names so partial help targets still work
    while remaining_tokens:
        next_token = remaining_tokens[0]
        if next_token.startswith("-"):
            break

        subparsers_action = find_subparsers_action(current_parser)
        if subparsers_action is None:
            break

        next_parser = subparsers_action.choices.get(next_token)
        if next_parser is None:
            break

        next_parser.prog = f"{current_parser.prog} {next_token}"
        current_parser = next_parser
        remaining_tokens.pop(0)

    return current_parser, remaining_tokens


def read_repl_line(
    prompt_text: str,
    input_stream: TextIO | None,
    output_stream: TextIO,
) -> str:
    """Read one REPL line from stdin or a provided stream."""

    if input_stream is None:
        return input(prompt_text)

    output_stream.write(prompt_text)
    output_stream.flush()
    next_line = input_stream.readline()
    if next_line == "":
        raise EOFError
    return next_line.rstrip("\r\n")


def print_repl_help(output_stream: TextIO) -> None:
    """Print the built-in REPL help."""

    output_stream.write("REPL commands:\n")
    output_stream.write("  commands               List available delegated commands\n")
    output_stream.write("  help                   Show root help\n")
    output_stream.write("  help <command...>      Show help for a specific delegated command\n")
    output_stream.write("  repl                   Show this help text again\n")
    output_stream.write("  exit | quit            Leave the REPL\n")
    output_stream.write("\n")
    output_stream.write("Run delegated commands exactly as you would from the CLI, for example:\n")
    output_stream.write("  preprocessing pipeline downloaded_ships --verbose\n")
    output_stream.write("  training build markov --graph-input-dir generated_ship_graphs_canonical\n")
    output_stream.write("  generator generate markov --output-dir out/generated-ships\n")


def run_repl(
    command_registry: dict[str, CommandDomain],
    root_parser: argparse.ArgumentParser,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
    error_stream: TextIO | None = None,
) -> int:
    """Run the interactive command loop."""

    resolved_output_stream = output_stream or sys.stdout
    resolved_error_stream = error_stream or sys.stderr

    resolved_output_stream.write(
        "Ship generator REPL\n"
        "Type 'commands' to list commands, 'help <command>' for help, and 'exit' to quit.\n"
    )

    while True:
        try:
            raw_command_line = read_repl_line(
                "ship-generator> ",
                input_stream=input_stream,
                output_stream=resolved_output_stream,
            )
        except EOFError:
            resolved_output_stream.write("\n")
            return 0
        except KeyboardInterrupt:
            resolved_output_stream.write("\n")
            continue

        stripped_command_line = raw_command_line.strip()
        if not stripped_command_line:
            continue

        try:
            command_tokens = split_repl_command_line(stripped_command_line)
        except ValueError as exc:
            resolved_error_stream.write(f"Command parse error: {exc}\n")
            continue

        first_token = command_tokens[0].lower()
        if first_token in {"exit", "quit"}:
            return 0
        if first_token == "commands":
            print_available_commands(command_registry, resolved_output_stream)
            continue
        if first_token in {"help", "repl"}:
            if len(command_tokens) == 1:
                if first_token == "repl":
                    print_repl_help(resolved_output_stream)
                else:
                    run_help_command((), command_registry, root_parser, resolved_output_stream)
                continue

            run_help_command(
                command_tokens[1:],
                command_registry,
                root_parser,
                resolved_output_stream,
            )
            continue

        exit_code = run_domain_command(
            command_tokens[0],
            command_tokens[1:],
            command_registry,
            error_stream=resolved_error_stream,
        )
        if exit_code != 0:
            resolved_error_stream.write(f"Command exited with status {exit_code}\n")


def run_cli(
    argv: Sequence[str],
    command_registry: dict[str, CommandDomain],
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
    error_stream: TextIO | None = None,
) -> int:
    """Run the root CLI or REPL from a provided argument sequence."""

    resolved_output_stream = output_stream or sys.stdout
    resolved_error_stream = error_stream or sys.stderr
    root_parser = build_root_parser(command_registry)
    parsed_args = root_parser.parse_args(list(argv))

    if parsed_args.entrypoint is None or parsed_args.entrypoint == "repl":
        return run_repl(
            command_registry,
            root_parser,
            input_stream=input_stream,
            output_stream=resolved_output_stream,
            error_stream=resolved_error_stream,
        )
    if parsed_args.entrypoint == "commands":
        print_available_commands(command_registry, resolved_output_stream)
        return 0
    if parsed_args.entrypoint == "help":
        return run_help_command(
            parsed_args.args,
            command_registry,
            root_parser,
            resolved_output_stream,
        )

    return run_domain_command(
        parsed_args.entrypoint,
        parsed_args.args,
        command_registry,
        error_stream=resolved_error_stream,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the unified ship generator entry point."""

    raw_arguments = list(argv) if argv is not None else sys.argv[1:]
    command_registry = build_domain_registry()
    return run_cli(raw_arguments, command_registry)


if __name__ == "__main__":
    raise SystemExit(main())
