"""Core symbolic math engine for MathNotebook using sympy."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import sympy as sp
from sympy import (
    symbols, Symbol, Rational, Integer, Float, I, pi, E, oo,
    sin, cos, tan, asin, acos, atan, sinh, cosh, tanh,
    exp, log, sqrt, Abs, sign,
    simplify, expand, factor, collect, apart, together,
    diff, integrate, limit, series,
    solve, Eq, solveset, linsolve, nonlinsolve,
    Matrix, zeros, ones, eye,
    Add, Mul, Pow, Function,
    srepr,
)
from sympy.abc import x, y, z, t
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application,
    convert_xor, function_exponentiation,
)
from sympy.physics.units import (
    UnitSystem, Dimension, Quantity,
    meter, kilogram, second, ampere, kelvin, mole, candela,
    newton, joule, watt, pascal, coulomb, volt, farad, ohm, siemens,
    weber, tesla, henry, lux, becquerel, gray, katal,
    hertz, radian, steradian,
)
from sympy.physics.units.systems import SI

TRANSFORMATIONS = (
    standard_transformations
    + (implicit_multiplication_application, convert_xor, function_exponentiation)
)

DEFAULT_SYMBOLS = {str(s): s for s in (x, y, z, t)}


@dataclass
class EngineState:
    """Mutable state for a notebook session."""
    symbols: dict[str, Symbol]
    functions: dict[str, sp.FunctionClass]
    assumptions: dict[str, Any]
    unit_system: UnitSystem

    def __init__(self):
        self.symbols = DEFAULT_SYMBOLS.copy()
        # Add common units to symbols for parsing
        self._add_units()
        self.functions = {}
        self.assumptions = {}
        self.unit_system = SI

    def _add_units(self):
        """Add sympy units to symbols for parsing."""
        from sympy.physics.units import (
            meter, kilogram, second, ampere, kelvin, mole, candela,
            newton, joule, watt, pascal, coulomb, volt, farad, ohm, siemens,
            weber, tesla, henry, lux, becquerel, gray, katal,
            hertz, radian, steradian,
        )
        units = {
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
        }
        self.symbols.update(units)

    def get_symbol(self, name: str, **assumptions) -> Symbol:
        if name in self.symbols:
            return self.symbols[name]
        sym = symbols(name, **assumptions)
        self.symbols[name] = sym
        return sym

    def get_symbols(self, names: str, **assumptions) -> tuple[Symbol, ...]:
        return tuple(self.get_symbol(n.strip(), **assumptions) for n in names.split())


class MathEngine:
    """Symbolic math engine with persistent state across evaluations."""

    def __init__(self):
        self.state = EngineState()

    def parse(self, expr_str: str) -> sp.Expr:
        """Parse a string expression into a sympy expression."""
        if not expr_str.strip():
            raise ValueError("Empty expression")
        return parse_expr(expr_str, transformations=TRANSFORMATIONS, local_dict=self.state.symbols)

    def evaluate(self, expr_str: str) -> sp.Expr:
        """Parse and evaluate an expression."""
        expr = self.parse(expr_str)
        return expr.evalf() if expr.has(Float) else expr

    def simplify_expr(self, expr_str: str) -> sp.Expr:
        return simplify(self.parse(expr_str))

    def expand_expr(self, expr_str: str) -> sp.Expr:
        return expand(self.parse(expr_str))

    def factor_expr(self, expr_str: str) -> sp.Expr:
        return factor(self.parse(expr_str))

    def collect_expr(self, expr_str: str, var_str: str) -> sp.Expr:
        expr = self.parse(expr_str)
        var = self.parse(var_str)
        return collect(expr, var)

    def apart_expr(self, expr_str: str, var_str: str) -> sp.Expr:
        expr = self.parse(expr_str)
        var = self.parse(var_str)
        return apart(expr, var)

    def together_expr(self, expr_str: str) -> sp.Expr:
        return together(self.parse(expr_str))

    # Calculus
    def diff_expr(self, expr_str: str, var_str: str, order: int = 1) -> sp.Expr:
        expr = self.parse(expr_str)
        var = self.parse(var_str)
        return diff(expr, var, order)

    def integrate_expr(self, expr_str: str, var_str: str, a: str | None = None, b: str | None = None) -> sp.Expr:
        expr = self.parse(expr_str)
        var = self.parse(var_str)
        if a is not None and b is not None:
            return integrate(expr, (var, self.parse(a), self.parse(b)))
        return integrate(expr, var)

    def limit_expr(self, expr_str: str, var_str: str, point_str: str, dir: str = "+-") -> sp.Expr:
        expr = self.parse(expr_str)
        var = self.parse(var_str)
        point = self.parse(point_str)
        return limit(expr, var, point, dir=dir)

    def series_expr(self, expr_str: str, var_str: str, point_str: str = "0", n: int = 6) -> sp.Expr:
        expr = self.parse(expr_str)
        var = self.parse(var_str)
        point = self.parse(point_str)
        return series(expr, var, point, n).removeO()

    # Equation solving
    def solve_eq(self, eq_str: str, var_str: str | None = None) -> list[sp.Expr]:
        expr = self.parse(eq_str)
        if var_str:
            var = self.parse(var_str)
            sols = solve(expr, var, dict=True)
            return [sol[var] for sol in sols if var in sol]
        else:
            return solve(expr, dict=True)

    def solve_system(self, eq_strs: list[str], var_strs: list[str]) -> list[dict]:
        eqs = [self.parse(e) for e in eq_strs]
        vars = [self.parse(v) for v in var_strs]
        return solve(eqs, vars, dict=True)

    def linsolve_system(self, eq_strs: list[str], var_strs: list[str]) -> sp.FiniteSet:
        eqs = [self.parse(e) for e in eq_strs]
        vars = [self.parse(v) for v in var_strs]
        return linsolve(eqs, vars)

    def nonlinsolve_system(self, eq_strs: list[str], var_strs: list[str]) -> sp.FiniteSet:
        eqs = [self.parse(e) for e in eq_strs]
        vars = [self.parse(v) for v in var_strs]
        return nonlinsolve(eqs, vars)

    # Matrix operations
    def matrix(self, rows: list[list[str]]) -> Matrix:
        return Matrix([[self.parse(cell) for cell in row] for row in rows])

    def det(self, matrix: Matrix) -> sp.Expr:
        return matrix.det()

    def inv(self, matrix: Matrix) -> Matrix:
        return matrix.inv()

    def eigenvalues(self, matrix: Matrix) -> list[sp.Expr]:
        return list(matrix.eigenvals().keys())

    def eigenvectors(self, matrix: Matrix) -> list[tuple[sp.Expr, int, list[Matrix]]]:
        return matrix.eigenvects()

    # Rational and exact arithmetic
    def rational(self, num: str, den: str = "1") -> Rational:
        return Rational(self.parse(num), self.parse(den))

    def nsimplify(self, expr_str: str, candidates: list[str] | None = None) -> sp.Expr:
        expr = self.parse(expr_str)
        return sp.nsimplify(expr, candidates=[self.parse(c) for c in candidates] if candidates else None)

    # Complex numbers
    def re(self, expr_str: str) -> sp.Expr:
        return sp.re(self.parse(expr_str))

    def im(self, expr_str: str) -> sp.Expr:
        return sp.im(self.parse(expr_str))

    def conjugate(self, expr_str: str) -> sp.Expr:
        return sp.conjugate(self.parse(expr_str))

    def arg(self, expr_str: str) -> sp.Expr:
        return sp.arg(self.parse(expr_str))

    # Units
    def with_units(self, value_str: str, unit_str: str) -> Quantity:
        value = self.parse(value_str)
        unit = self.parse_unit(unit_str)
        return Quantity(value, unit)

    def parse_unit(self, unit_str: str):
        unit_str = unit_str.strip()
        unit_map = {
            "m": meter, "meter": meter, "meters": meter,
            "kg": kilogram, "kilogram": kilogram, "kilograms": kilogram,
            "s": second, "second": second, "seconds": second,
            "A": ampere, "ampere": ampere, "amperes": ampere,
            "K": kelvin, "kelvin": kelvin,
            "mol": mole, "mole": mole,
            "cd": candela, "candela": candela,
            "N": newton, "newton": newton,
            "J": joule, "joule": joule,
            "W": watt, "watt": watt,
            "Pa": pascal, "pascal": pascal,
            "C": coulomb, "coulomb": coulomb,
            "V": volt, "volt": volt,
            "F": farad, "farad": farad,
            "ohm": ohm, "Ω": ohm,
            "S": siemens, "siemens": siemens,
            "Wb": weber, "weber": weber,
            "T": tesla, "tesla": tesla,
            "H": henry, "henry": henry,
            "lx": lux, "lux": lux,
            "Bq": becquerel, "becquerel": becquerel,
            "Gy": gray, "gray": gray,
            
            "kat": katal, "katal": katal,
            "Hz": hertz, "hertz": hertz,
            "rad": radian, "radian": radian,
            "sr": steradian, "steradian": steradian,
        }
        unit_str_lower = unit_str.lower()
        for key, unit in unit_map.items():
            if key.lower() == unit_str_lower:
                return unit
        try:
            return self.parse(unit_str)
        except Exception:
            raise ValueError(f"Unknown unit: {unit_str}")

    def convert_unit(self, quantity: Quantity, target_unit_str: str) -> Quantity:
        target_unit = self.parse_unit(target_unit_str)
        return quantity.convert_to(target_unit)

    def simplify_units(self, expr_str: str) -> sp.Expr:
        expr = self.parse(expr_str)
        return sp.simplify(expr)

    # LaTeX output
    def to_latex(self, expr_str: str) -> str:
        expr = self.parse(expr_str)
        return sp.latex(expr)

    def to_pretty(self, expr_str: str) -> str:
        expr = self.parse(expr_str)
        return sp.pretty(expr, use_unicode=True)

    def to_srepr(self, expr_str: str) -> str:
        expr = self.parse(expr_str)
        return srepr(expr)

    # Define symbols and functions
    def define_symbol(self, name: str, **assumptions) -> Symbol:
        return self.state.get_symbol(name, **assumptions)

    def define_function(self, name: str, expr_str: str, args: list[str]) -> sp.FunctionClass:
        f = Function(name)
        arg_symbols = [self.state.get_symbol(a) for a in args]
        expr = self.parse(expr_str)
        self.state.functions[name] = f
        self.state.symbols[name] = f
        return f

    def substitute(self, expr_str: str, subs: dict[str, str]) -> sp.Expr:
        expr = self.parse(expr_str)
        sub_dict = {self.parse(k): self.parse(v) for k, v in subs.items()}
        return expr.subs(sub_dict)

    def to_numeric(self, expr_str: str) -> float:
        expr = self.parse(expr_str)
        return float(expr.evalf())


ENGINE = MathEngine()