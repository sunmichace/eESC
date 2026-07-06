from __future__ import annotations

"""Unified command-line entry points for the eESC project."""

import argparse
import sys

from eesc.analysis import stacking, training_history
from eesc import legacy
from eesc import task_views


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    metrics_parser = subparsers.add_parser(
        "metrics",
        help="Generate task-level performance figures from stacking logs.",
    )
    stacking.add_arguments(metrics_parser)
    metrics_parser.set_defaults(handler=stacking.run_from_args)

    training_parser = subparsers.add_parser(
        "training-r2",
        help="Generate repeated-training R2 history figures.",
    )
    training_history.add_arguments(training_parser)
    training_parser.set_defaults(handler=training_history.run_from_args)

    tasks_parser = subparsers.add_parser(
        "tasks",
        help="List standardized task metadata.",
    )
    legacy.add_list_tasks_arguments(tasks_parser)
    tasks_parser.set_defaults(handler=legacy.run_list_tasks)

    task_view_parser = subparsers.add_parser(
        "task-view",
        help="Show the standardized view of a task branch.",
    )
    task_view_parser.add_argument("task", choices=tuple(task_views.TASK_VIEWS))
    task_view_parser.set_defaults(handler=task_views.run_task_view)

    task_run_parser = subparsers.add_parser(
        "task-run",
        help="Run a task through a stable alias instead of a raw underlying script path.",
    )
    task_run_parser.add_argument("task", choices=tuple(task_views.TASK_VIEWS))
    task_run_parser.add_argument("alias", help="Stable entrypoint alias defined for the task.")
    task_run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without executing it.",
    )
    task_run_parser.add_argument(
        "--show-template",
        action="store_true",
        help="Print the recommended command template and configuration hints for this alias.",
    )
    task_run_parser.set_defaults(handler=task_views.run_task_entrypoint)

    return parser


def _split_passthrough_args(argv: list[str]) -> tuple[list[str], list[str]]:
    if argv and argv[0] == "task-run" and "--" in argv:
        separator_index = argv.index("--")
        return argv[:separator_index], argv[separator_index + 1 :]
    return argv, []


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    argv, passthrough_args = _split_passthrough_args(argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) == "task-run":
        args.script_args = passthrough_args
    args.handler(args)


if __name__ == "__main__":
    main()
