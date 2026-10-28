"""Command line for PhAST result files: ``phast result <command>``.

Commands: ``validate``, ``recover`` and ``finalize``. Further commands are added by registering them with
``_register``.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Callable, Sequence

from .finalize import finalize
from .recover import recover
from .validate import validate


def _validate(args: argparse.Namespace) -> int:
    report = validate(args.file, allow_running=args.running_ok)
    print(json.dumps(report.to_dict(), indent=1) if args.json else report.format_text())
    return 0 if report.ok else 1


def _recover(args: argparse.Namespace) -> int:
    try:
        result = recover(args.file, args.output, overwrite=args.force)
    except (OSError, ValueError, KeyError) as exc:
        print(f"recover failed: {exc}", file=sys.stderr)
        return 1
    print(result.summary())
    print(result.report.format_text())
    return 0 if result.report.ok else 1


def _finalize(args: argparse.Namespace) -> int:
    if not args.archive:
        print("nothing to do: pass --archive to write the archive copy", file=sys.stderr)
        return 1
    try:
        path = finalize(args.file, archive=True)
    except (OSError, RuntimeError) as exc:
        print(f"finalize failed: {exc}", file=sys.stderr)
        return 1
    report = validate(path)
    print(f"archived {path.name} (gzip 4, shuffle, fletcher32)")
    print(report.format_text())
    return 0 if report.ok else 1


# name -> (help, add_arguments, handler); new commands are registered here
COMMANDS: dict[str, tuple[str, Callable[[argparse.ArgumentParser], None], Callable[[argparse.Namespace], int]]] = {}


def _register(name: str, help_text: str, add: Callable[[argparse.ArgumentParser], None],
              run: Callable[[argparse.Namespace], int]) -> None:
    COMMANDS[name] = (help_text, add, run)


def _validate_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("file", help="result file (.vtkhdf)")
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    p.add_argument("--running-ok", action="store_true",
                   help="accept status = running (a file on scratch whose run is still active)")


def _recover_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("file", help="interrupted result file; it is not modified")
    p.add_argument("-o", "--output", default=None, help="output path (default: <stem>.recovered.vtkhdf beside the file)")
    p.add_argument("--force", action="store_true", help="replace the output file if it exists")


def _finalize_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("file", help="closed result file (.vtkhdf); replaced by its checked archive copy")
    p.add_argument("--archive", action="store_true", help="rewrite with gzip 4, shuffle and fletcher32")


_register("validate", "check a result file against layout 2.x; exit 1 if any ERROR", _validate_args, _validate)
_register("recover", "write a trimmed, validated copy of an interrupted result file", _recover_args, _recover)
_register("finalize", "rewrite a closed result file with the archive filters and check it", _finalize_args, _finalize)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phast result", description="Check and repair PhAST result files (layout 2.x).")
    sub = parser.add_subparsers(dest="command", metavar="command", required=True)
    for name, (help_text, add, run) in COMMANDS.items():
        p = sub.add_parser(name, help=help_text, description=help_text)
        add(p)
        p.set_defaults(handler=run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
