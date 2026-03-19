"""Tests for the unified root CLI entry point."""

from __future__ import annotations

import argparse
from io import StringIO

import main


def build_stub_parser() -> argparse.ArgumentParser:
    """Build a parser with nested subcommands for registry tests."""

    parser = argparse.ArgumentParser(prog="stub")
    action_subparsers = parser.add_subparsers(dest="action", required=True)
    build_parser = action_subparsers.add_parser("build")
    build_backend_subparsers = build_parser.add_subparsers(dest="backend", required=True)
    build_backend_subparsers.add_parser("alpha")
    action_subparsers.add_parser("validate")
    return parser


def test_iter_leaf_command_paths_discovers_nested_commands() -> None:
    """Command discovery should include every leaf command path."""

    discovered_paths = list(main.iter_leaf_command_paths(build_stub_parser()))
    assert discovered_paths == [("build", "alpha"), ("validate",)]


def test_build_domain_registry_exposes_current_top_level_domains() -> None:
    """The root registry should include the current pipeline domains."""

    command_registry = main.build_domain_registry()
    assert set(command_registry) == {"corpus", "generator", "graph-expansion", "preprocessing", "training", "visualizer"}


def test_run_cli_delegates_to_selected_domain() -> None:
    """The root CLI should pass through arguments to the chosen domain."""

    captured_arguments: list[str] = []

    def run_stub(command_arguments: list[str] | None) -> int:
        """Capture delegated arguments for assertion."""

        assert command_arguments is not None
        captured_arguments.extend(command_arguments)
        return 0

    command_registry = {
        "training": main.CommandDomain(
            name="training",
            help_text="Stub training domain",
            build_parser=build_stub_parser,
            run=run_stub,
        )
    }

    exit_code = main.run_cli(["training", "build", "alpha"], command_registry)

    assert exit_code == 0
    assert captured_arguments == ["build", "alpha"]


def test_split_repl_command_line_strips_windows_quotes() -> None:
    """Quoted Windows-style REPL paths should be unwrapped correctly."""

    command_tokens = main.split_repl_command_line(
        'generator generate markov --output-dir "out/generated ships"'
    )

    assert command_tokens == [
        "generator",
        "generate",
        "markov",
        "--output-dir",
        "out/generated ships",
    ]


def test_run_help_command_renders_child_parser_to_provided_stream() -> None:
    """Delegated help should use the resolved child parser and captured output."""

    root_parser = main.build_root_parser({})
    root_parser.prog = "main.py"
    output_stream = StringIO()

    def fail_if_called(command_arguments: list[str] | None) -> int:
        """Help rendering should not dispatch through the delegated runner."""

        raise AssertionError("Delegated help should not execute the command runner")

    command_registry = {
        "training": main.CommandDomain(
            name="training",
            help_text="Stub training domain",
            build_parser=build_stub_parser,
            run=fail_if_called,
        )
    }

    exit_code = main.run_help_command(
        ["training", "build", "alpha"],
        command_registry,
        root_parser,
        output_stream,
    )

    rendered_output = output_stream.getvalue()

    assert exit_code == 0
    assert "usage: main.py training build alpha" in rendered_output
    assert "usage: stub build alpha" not in rendered_output


def test_run_repl_supports_listing_help_and_quit() -> None:
    """The REPL should handle built-in commands without crashing."""

    input_stream = StringIO("commands\nhelp training build alpha\nquit\n")
    output_stream = StringIO()
    error_stream = StringIO()

    command_registry = {
        "training": main.CommandDomain(
            name="training",
            help_text="Stub training domain",
            build_parser=build_stub_parser,
            run=lambda command_arguments: 0,
        )
    }

    exit_code = main.run_repl(
        command_registry,
        main.build_root_parser(command_registry),
        input_stream=input_stream,
        output_stream=output_stream,
        error_stream=error_stream,
    )

    rendered_output = output_stream.getvalue()

    assert exit_code == 0
    assert "Voidwright REPL" in rendered_output
    assert "training build alpha" in rendered_output
    assert error_stream.getvalue() == ""
