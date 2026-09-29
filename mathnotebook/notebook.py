"""Notebook data model with reactive evaluation and dependency tracking."""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any

import sympy as sp
from mathnotebook.engine import ENGINE, MathEngine


class CellType(Enum):
    CODE = "code"
    MARKDOWN = "markdown"
    RAW = "raw"


class CellStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"


@dataclass
class CellOutput:
    output_type: str
    data: dict[str, str]
    metadata: dict = field(default_factory=dict)


@dataclass
class Cell:
    id: str
    cell_type: CellType
    source: str
    outputs: list[CellOutput] = field(default_factory=list)
    status: CellStatus = CellStatus.IDLE
    execution_count: int | None = None
    metadata: dict = field(default_factory=dict)
    dependencies: set[str] = field(default_factory=set)
    dependents: set[str] = field(default_factory=set)
    defined_names: set[str] = field(default_factory=set)
    referenced_names: set[str] = field(default_factory=set)

    @staticmethod
    def new_code(source: str = "") -> "Cell":
        return Cell(
            id=str(uuid.uuid4())[:8],
            cell_type=CellType.CODE,
            source=source,
        )

    @staticmethod
    def new_markdown(source: str = "") -> "Cell":
        return Cell(
            id=str(uuid.uuid4())[:8],
            cell_type=CellType.MARKDOWN,
            source=source,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cell_type"] = self.cell_type.value
        d["status"] = self.status.value
        d["dependencies"] = list(self.dependencies)
        d["dependents"] = list(self.dependents)
        d["defined_names"] = list(self.defined_names)
        d["referenced_names"] = list(self.referenced_names)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Cell":
        data = data.copy()
        data["cell_type"] = CellType(data["cell_type"])
        data["status"] = CellStatus(data["status"])
        data["dependencies"] = set(data.get("dependencies", []))
        data["dependents"] = set(data.get("dependents", []))
        data["defined_names"] = set(data.get("defined_names", []))
        data["referenced_names"] = set(data.get("referenced_names", []))
        outputs = []
        for out in data.get("outputs", []):
            outputs.append(CellOutput(**out))
        data["outputs"] = outputs
        return cls(**data)


class DependencyAnalyzer:
    """Analyzes cell source code to extract defined and referenced names."""

    ASSIGNMENT_PATTERN = re.compile(r'^(\w+)\s*=')
    FUNCTION_DEF_PATTERN = re.compile(r'^def\s+(\w+)\s*\(')
    CLASS_DEF_PATTERN = re.compile(r'^class\s+(\w+)')
    IMPORT_PATTERN = re.compile(r'^(?:from\s+\S+\s+)?import\s+(.+)$')

    def __init__(self, engine: MathEngine):
        self.engine = engine

    def analyze(self, source: str) -> tuple[set[str], set[str]]:
        """Return (defined_names, referenced_names) from source code."""
        defined = set()
        referenced = set()

        for line in source.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Check for assignments
            m = self.ASSIGNMENT_PATTERN.match(line)
            if m:
                defined.add(m.group(1))
                # Also extract referenced names from RHS
                rhs = line[m.end():].strip()
                if rhs:
                    try:
                        expr = self.engine.parse(rhs)
                        for sym in expr.free_symbols:
                            referenced.add(str(sym))
                    except Exception:
                        words = re.findall(r'\b[a-zA-Z_]\w*\b', rhs)
                        for word in words:
                            if word not in ("def", "class", "import", "from", "as", "if", "else", "for", "while", "try", "except", "return", "True", "False", "None"):
                                referenced.add(word)
                continue

            # Check for function definitions
            m = self.FUNCTION_DEF_PATTERN.match(line)
            if m:
                defined.add(m.group(1))
                continue

            # Check for class definitions
            m = self.CLASS_DEF_PATTERN.match(line)
            if m:
                defined.add(m.group(1))
                continue

            # Check for imports
            m = self.IMPORT_PATTERN.match(line)
            if m:
                imports = m.group(1).replace(" ", "").split(",")
                for imp in imports:
                    name = imp.split(".")[-1]
                    if " as " in name:
                        name = name.split(" as ")[-1]
                    defined.add(name)
                continue

            # For other lines, extract referenced names from sympy symbols
            try:
                expr = self.engine.parse(line)
                for sym in expr.free_symbols:
                    referenced.add(str(sym))
            except Exception:
                # If parsing fails, fall back to simple word extraction
                words = re.findall(r'\b[a-zA-Z_]\w*\b', line)
                for word in words:
                    if word not in ("def", "class", "import", "from", "as", "if", "else", "for", "while", "try", "except", "return", "True", "False", "None"):
                        referenced.add(word)

        return defined, referenced


class CycleDetector:
    """Detects cycles in the cell dependency graph."""

    @staticmethod
    def has_cycle(cells: dict[str, Cell], start_id: str, new_deps: set[str]) -> tuple[bool, list[str]]:
        """Check if adding new_deps to start_id would create a cycle."""
        # Build adjacency list
        graph = {cid: set(cell.dependencies) for cid, cell in cells.items()}
        graph[start_id] = graph.get(start_id, set()) | new_deps

        # DFS to detect cycle
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node: str):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, set()):
                if neighbor not in cells:
                    continue
                if neighbor not in visited:
                    result = dfs(neighbor)
                    if result:
                        return result
                elif neighbor in rec_stack:
                    # Found cycle - build the cycle path
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:] + [neighbor]

            rec_stack.remove(node)
            path.pop()
            return False

        result = dfs(start_id)
        if isinstance(result, list):
            return True, result
        return False, []

    @staticmethod
    def topological_sort(cells: dict[str, Cell]) -> list[str]:
        """Return cells in topological order (dependencies first)."""
        graph = {cid: set(cell.dependencies) for cid, cell in cells.items()}
        in_degree = {cid: 0 for cid in cells}
        for deps in graph.values():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] += 1

        queue = [cid for cid, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for cell in cells.values():
                if node in cell.dependencies:
                    in_degree[cell.id] -= 1
                    if in_degree[cell.id] == 0:
                        queue.append(cell.id)

        if len(result) != len(cells):
            # Cycle exists
            remaining = set(cells.keys()) - set(result)
            return result + list(remaining)

        return result


class Notebook:
    """Reactive notebook with cell dependency tracking."""

    def __init__(self, engine: MathEngine | None = None):
        self.engine = engine or ENGINE
        self.cells: dict[str, Cell] = {}
        self.cell_order: list[str] = []
        self.execution_counter = 0
        self.analyzer = DependencyAnalyzer(self.engine)
        self.metadata = {
            "created": datetime.now().isoformat(),
            "modified": datetime.now().isoformat(),
            "version": "1.0",
        }

    def add_cell(self, cell: Cell, index: int | None = None) -> str:
        """Add a cell to the notebook."""
        self.cells[cell.id] = cell
        if index is None:
            self.cell_order.append(cell.id)
        else:
            self.cell_order.insert(index, cell.id)
        self._update_dependencies(cell)
        self.metadata["modified"] = datetime.now().isoformat()
        return cell.id

    def remove_cell(self, cell_id: str) -> bool:
        """Remove a cell and update dependencies."""
        if cell_id not in self.cells:
            return False
        cell = self.cells[cell_id]
        # Remove from dependents' dependencies
        for dep_id in cell.dependents:
            if dep_id in self.cells:
                self.cells[dep_id].dependencies.discard(cell_id)
        # Remove from dependencies' dependents
        for dep_id in cell.dependencies:
            if dep_id in self.cells:
                self.cells[dep_id].dependents.discard(cell_id)
        del self.cells[cell_id]
        self.cell_order = [cid for cid in self.cell_order if cid != cell_id]
        self.metadata["modified"] = datetime.now().isoformat()
        return True

    def move_cell(self, cell_id: str, new_index: int) -> bool:
        """Move a cell to a new position."""
        if cell_id not in self.cells:
            return False
        self.cell_order.remove(cell_id)
        self.cell_order.insert(new_index, cell_id)
        self.metadata["modified"] = datetime.now().isoformat()
        return True

    def get_cell(self, cell_id: str) -> Cell | None:
        return self.cells.get(cell_id)

    def _update_dependencies(self, cell: Cell) -> None:
        """Update dependency graph for a cell based on its source."""
        if cell.cell_type != CellType.CODE:
            return

        defined, referenced = self.analyzer.analyze(cell.source)
        cell.defined_names = defined
        cell.referenced_names = referenced

        # Find which cells define the referenced names
        new_deps = set()
        for ref in referenced:
            for other_id, other_cell in self.cells.items():
                if other_id == cell.id:
                    continue
                if ref in other_cell.defined_names:
                    new_deps.add(other_id)

        # Check for cycles
        has_cycle, cycle_path = CycleDetector.has_cycle(self.cells, cell.id, new_deps)
        if has_cycle:
            # Don't update dependencies if it creates a cycle
            cell.outputs = [CellOutput(
                output_type="error",
                data={"text/plain": f"Cycle detected: {' -> '.join(cycle_path)}"},
            )]
            cell.status = CellStatus.ERROR
            return

        # Update dependencies
        old_deps = cell.dependencies
        cell.dependencies = new_deps

        # Update reverse dependencies
        for dep_id in old_deps - new_deps:
            if dep_id in self.cells:
                self.cells[dep_id].dependents.discard(cell.id)
        for dep_id in new_deps - old_deps:
            if dep_id in self.cells:
                self.cells[dep_id].dependents.add(cell.id)

        # Also update other cells that reference names this cell defines
        # This handles the case where a new cell defines a name that existing cells need
        for defined_name in cell.defined_names:
            for other_id, other_cell in self.cells.items():
                if other_id == cell.id:
                    continue
                if defined_name in other_cell.referenced_names:
                    # Other cell references this name, so it depends on this cell
                    if cell.id not in other_cell.dependencies:
                        # Check for cycle
                        has_cycle, cycle_path = CycleDetector.has_cycle(
                            self.cells, other_id, other_cell.dependencies | {cell.id}
                        )
                        if not has_cycle:
                            other_cell.dependencies.add(cell.id)
                            cell.dependents.add(other_id)
                        else:
                            # Mark the other cell as having a cycle error
                            other_cell.outputs = [CellOutput(
                                output_type="error",
                                data={"text/plain": f"Cycle detected: {' -> '.join(cycle_path)}"},
                            )]
                            other_cell.status = CellStatus.ERROR

    def evaluate_cell(self, cell_id: str) -> Cell:
        """Evaluate a single cell and its dependencies."""
        cell = self.cells.get(cell_id)
        if not cell:
            raise ValueError(f"Cell {cell_id} not found")

        if cell.cell_type != CellType.CODE:
            cell.status = CellStatus.SUCCESS
            return cell

        # Evaluate dependencies first
        eval_order = self._get_evaluation_order(cell_id)
        for dep_id in eval_order:
            if dep_id == cell_id:
                continue
            dep_cell = self.cells[dep_id]
            # Check if dependency already has an error
            if dep_cell.status == CellStatus.ERROR:
                cell.status = CellStatus.ERROR
                cell.outputs = [CellOutput(
                    output_type="error",
                    data={"text/plain": f"Dependency error in {dep_id[:8]}: {dep_cell.outputs[0].data.get('text/plain', 'Unknown error')}"},
                )]
                return cell
            self.evaluate_cell(dep_id)
            # Check if dependency evaluation resulted in an error
            if dep_cell.status == CellStatus.ERROR:
                cell.status = CellStatus.ERROR
                cell.outputs = [CellOutput(
                    output_type="error",
                    data={"text/plain": f"Dependency error in {dep_id[:8]}: {dep_cell.outputs[0].data.get('text/plain', 'Unknown error')}"},
                )]
                return cell

        # Now evaluate this cell
        return self._execute_cell(cell)

    def _get_evaluation_order(self, cell_id: str) -> list[str]:
        """Get cells in dependency order for evaluation."""
        visited = set()
        order = []

        def visit(cid: str):
            if cid in visited or cid not in self.cells:
                return
            visited.add(cid)
            for dep in self.cells[cid].dependencies:
                visit(dep)
            order.append(cid)

        visit(cell_id)
        return order

    def _get_exec_namespace(self) -> dict:
        """Get the execution namespace with sympy functions and engine symbols."""
        from sympy.physics.units import (
            meter, kilogram, second, ampere, kelvin, mole, candela,
            newton, joule, watt, pascal, coulomb, volt, farad, ohm, siemens,
            weber, tesla, henry, lux, becquerel, gray, katal,
            hertz, radian, steradian, convert_to,
            kilometer, gram, minute, hour, day, liter, bar, atm,
            electronvolt, angstrom, foot, inch, mile, yard, pound, psi,
        )
        ns = dict(self.engine.state.symbols)
        ns.update({
            "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
            "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
            "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
            "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
            "Abs": sp.Abs, "sign": sp.sign,
            "simplify": sp.simplify, "expand": sp.expand, "factor": sp.factor,
            "collect": sp.collect, "apart": sp.apart, "together": sp.together,
            "diff": sp.diff, "integrate": sp.integrate, "limit": sp.limit, "series": sp.series,
            "solve": sp.solve, "Eq": sp.Eq, "solveset": sp.solveset,
            "linsolve": sp.linsolve, "nonlinsolve": sp.nonlinsolve,
            "Matrix": sp.Matrix, "zeros": sp.zeros, "ones": sp.ones, "eye": sp.eye,
            "Rational": sp.Rational, "Integer": sp.Integer, "Float": sp.Float,
            "I": sp.I, "pi": sp.pi, "E": sp.E, "oo": sp.oo,
            "meter": meter, "kilogram": kilogram, "second": second,
            "ampere": ampere, "kelvin": kelvin, "mole": mole, "candela": candela,
            "newton": newton, "joule": joule, "watt": watt, "pascal": pascal,
            "coulomb": coulomb, "volt": volt, "farad": farad, "ohm": ohm, "siemens": siemens,
            "weber": weber, "tesla": tesla, "henry": henry, "lux": lux,
            "becquerel": becquerel, "gray": gray, "katal": katal,
            "hertz": hertz, "radian": radian, "steradian": steradian,
            "m": meter, "kg": kilogram, "s": second, "A": ampere, "K": kelvin,
            "mol": mole, "cd": candela, "N": newton, "J": joule, "W": watt,
            "Pa": pascal, "C": coulomb, "V": volt, "F": farad, "S": siemens,
            "Wb": weber, "T": tesla, "H": henry, "Hz": hertz, "rad": radian,
            "km": kilometer, "kilometer": kilometer, "hour": hour,
            "g": gram, "min": minute, "hr": hour, "day": day, "L": liter,
            "bar": bar, "atm": atm, "eV": electronvolt, "Å": angstrom,
            "ft": foot, "in": inch, "mi": mile, "yd": yard, "lb": pound, "psi": psi,
            "convert": convert_to,
        })
        return ns

    def _execute_cell(self, cell: Cell) -> Cell:
        """Execute a code cell."""
        self.execution_counter += 1
        cell.execution_count = self.execution_counter
        cell.status = CellStatus.RUNNING
        cell.outputs = []

        try:
            # Execute in the engine's context
            # We need to capture the result of the last expression
            source = cell.source.strip()
            if not source:
                cell.status = CellStatus.SUCCESS
                return cell

            # Split into lines and execute each
            lines = source.splitlines()
            last_result = None
            exec_ns = self._get_exec_namespace()

            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                try:
                    # Try to parse as expression first
                    expr = self.engine.parse(line)
                    # Check if it's a sympy expression (has the 'has' method)
                    if hasattr(expr, 'has'):
                        result = expr.evalf() if expr.has(sp.Float) else expr
                    else:
                        # parse_expr can return lists (e.g., from solve)
                        result = expr
                    last_result = result
                    # Update namespace with any new symbols
                    exec_ns.update(self.engine.state.symbols)
                except Exception:
                    # If not an expression, try eval (for function calls that return values)
                    try:
                        last_result = eval(line, exec_ns, exec_ns)
                    except Exception:
                        # Last resort: exec it (for assignments, function defs, etc.)
                        exec(line, exec_ns, exec_ns)
                    # Update engine symbols from exec namespace
                    # Sync any new variables (including non-sympy values)
                    for k, v in exec_ns.items():
                        if k.startswith('_'):
                            continue
                        if k not in self.engine.state.symbols:
                            # New variable
                            if isinstance(v, (int, float, complex, str)):
                                self.engine.state.symbols[k] = sp.sympify(v)
                            elif isinstance(v, sp.Basic):
                                self.engine.state.symbols[k] = v
                            elif callable(v):
                                self.engine.state.symbols[k] = v
                        else:
                            # Existing symbol - update if it has a concrete value
                            existing = self.engine.state.symbols[k]
                            if isinstance(existing, sp.Symbol) and not isinstance(existing, (sp.Number, sp.Float)):
                                # It's a plain symbol, replace with value
                                if isinstance(v, (int, float, complex, str)):
                                    self.engine.state.symbols[k] = sp.sympify(v)
                                elif isinstance(v, sp.Basic) and not isinstance(v, sp.Symbol):
                                    self.engine.state.symbols[k] = v

            # Record output
            if last_result is not None:
                cell.outputs.append(CellOutput(
                    output_type="execute_result",
                    data={
                        "text/plain": str(last_result),
                        "text/latex": self.engine.to_latex(str(last_result)),
                    },
                ))

            cell.status = CellStatus.SUCCESS

        except Exception as e:
            cell.status = CellStatus.ERROR
            cell.outputs.append(CellOutput(
                output_type="error",
                data={"text/plain": f"{type(e).__name__}: {e}"},
            ))

        return cell

    def evaluate_all(self) -> list[Cell]:
        """Evaluate all code cells in topological order."""
        sorted_ids = CycleDetector.topological_sort(self.cells)
        results = []
        for cid in sorted_ids:
            cell = self.cells[cid]
            if cell.cell_type == CellType.CODE:
                results.append(self.evaluate_cell(cid))
        return results

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata,
            "cells": [self.cells[cid].to_dict() for cid in self.cell_order],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict, engine: MathEngine | None = None) -> "Notebook":
        nb = cls(engine)
        nb.metadata = data.get("metadata", nb.metadata)
        for cell_data in data.get("cells", []):
            cell = Cell.from_dict(cell_data)
            nb.cells[cell.id] = cell
            nb.cell_order.append(cell.id)
        # Rebuild dependencies
        for cell in nb.cells.values():
            nb._update_dependencies(cell)
        return nb

    @classmethod
    def from_json(cls, json_str: str, engine: MathEngine | None = None) -> "Notebook":
        return cls.from_dict(json.loads(json_str), engine)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str, engine: MathEngine | None = None) -> "Notebook":
        with open(path) as f:
            return cls.from_json(f.read(), engine)