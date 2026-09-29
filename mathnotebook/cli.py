"""Command-line interface for MathNotebook."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mathnotebook.engine import ENGINE, MathEngine
from mathnotebook.notebook import Cell, CellType, Notebook


def cmd_new(args: argparse.Namespace) -> int:
    nb = Notebook(ENGINE)
    nb.save(args.output)
    print(f"Created new notebook: {args.output}")
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    nb = Notebook.load(args.notebook, ENGINE)
    if args.type == "code":
        cell = Cell.new_code(args.source)
    elif args.type == "markdown":
        cell = Cell.new_markdown(args.source)
    else:
        cell = Cell.new_code(args.source)
    nb.add_cell(cell, args.index)
    nb.save(args.notebook)
    print(f"Added {args.type} cell: {cell.id}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    nb = Notebook.load(args.notebook, ENGINE)
    if args.cell:
        cell = nb.evaluate_cell(args.cell)
        print(f"Cell {cell.id}: {cell.status.value}")
        for out in cell.outputs:
            print(out.data.get("text/plain", ""))
    else:
        results = nb.evaluate_all()
        for cell in results:
            print(f"Cell {cell.id}: {cell.status.value}")
            for out in cell.outputs:
                print(out.data.get("text/plain", ""))
    nb.save(args.notebook)
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    nb = Notebook.load(args.notebook, ENGINE)
    for i, cid in enumerate(nb.cell_order):
        cell = nb.cells[cid]
        print(f"[{i}] {cid} ({cell.cell_type.value}) - {cell.status.value}")
        if args.source:
            print(f"    {cell.source[:80]}...")
        if cell.outputs:
            for out in cell.outputs:
                print(f"    -> {out.data.get('text/plain', '')[:80]}")
    return 0


def cmd_repl(args: argparse.Namespace) -> int:
    nb = Notebook.load(args.notebook, ENGINE) if args.notebook else Notebook(ENGINE)
    print("MathNotebook REPL (type 'exit' to quit, 'save' to save, 'eval' to evaluate)")
    while True:
        try:
            line = input("In [ ]: ")
        except EOFError:
            break

        if line.strip() == "exit":
            break
        if line.strip() == "save":
            if args.notebook:
                nb.save(args.notebook)
                print(f"Saved to {args.notebook}")
            else:
                print("No notebook file specified. Use --notebook to save.")
            continue
        if line.strip() == "eval":
            results = nb.evaluate_all()
            for cell in results:
                print(f"Cell {cell.id}: {cell.status.value}")
                for out in cell.outputs:
                    print(out.data.get("text/plain", ""))
            continue
        if line.strip() == "show":
            for i, cid in enumerate(nb.cell_order):
                cell = nb.cells[cid]
                print(f"[{i}] {cid} ({cell.cell_type.value}) - {cell.status.value}")
                print(f"    {cell.source[:80]}")
            continue

        cell = Cell.new_code(line)
        nb.add_cell(cell)
        nb.evaluate_cell(cell.id)
        for out in cell.outputs:
            print(out.data.get("text/plain", ""))

    if args.notebook:
        nb.save(args.notebook)
        print(f"Saved to {args.notebook}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="mathnb", description="MathNotebook CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # new
    p_new = subparsers.add_parser("new", help="Create a new notebook")
    p_new.add_argument("output", help="Output .mathnb file")
    p_new.set_defaults(func=cmd_new)

    # add
    p_add = subparsers.add_parser("add", help="Add a cell to a notebook")
    p_add.add_argument("notebook", help="Notebook file")
    p_add.add_argument("source", help="Cell source code")
    p_add.add_argument("--type", choices=["code", "markdown"], default="code", help="Cell type")
    p_add.add_argument("--index", type=int, help="Insert at index")
    p_add.set_defaults(func=cmd_add)

    # eval
    p_eval = subparsers.add_parser("eval", help="Evaluate notebook or cell")
    p_eval.add_argument("notebook", help="Notebook file")
    p_eval.add_argument("--cell", help="Specific cell ID to evaluate")
    p_eval.set_defaults(func=cmd_eval)

    # show
    p_show = subparsers.add_parser("show", help="Show notebook contents")
    p_show.add_argument("notebook", help="Notebook file")
    p_show.add_argument("--source", action="store_true", help="Show cell source")
    p_show.set_defaults(func=cmd_show)

    # repl
    p_repl = subparsers.add_parser("repl", help="Interactive REPL")
    p_repl.add_argument("--notebook", help="Notebook file to load/save")
    p_repl.set_defaults(func=cmd_repl)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())