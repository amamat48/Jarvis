"""Tests for MathNotebook."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mathnotebook.engine import MathEngine
from mathnotebook.notebook import Cell, CellType, Notebook


class MathNotebookTests(unittest.TestCase):
    def setUp(self):
        self.engine = MathEngine()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_basic_arithmetic(self):
        nb = Notebook(self.engine)
        cell = Cell.new_code("2 + 2")
        nb.add_cell(cell)
        result = nb.evaluate_cell(cell.id)
        self.assertEqual(result.status.value, "success")
        self.assertEqual(len(result.outputs), 1)
        self.assertEqual(result.outputs[0].data["text/plain"], "4")

    def test_variable_assignment_and_use(self):
        nb = Notebook(self.engine)
        cell1 = Cell.new_code("x = 5")
        cell2 = Cell.new_code("x * 3")
        nb.add_cell(cell1)
        nb.add_cell(cell2)
        result2 = nb.evaluate_cell(cell2.id)
        self.assertEqual(result2.status.value, "success")
        self.assertEqual(result2.outputs[0].data["text/plain"], "15")

    def test_symbolic_simplification(self):
        nb = Notebook(self.engine)
        cell = Cell.new_code("simplify(sin(x)**2 + cos(x)**2)")
        nb.add_cell(cell)
        result = nb.evaluate_cell(cell.id)
        self.assertEqual(result.status.value, "success")
        self.assertEqual(result.outputs[0].data["text/plain"], "1")

    def test_derivative(self):
        nb = Notebook(self.engine)
        cell = Cell.new_code("diff(sin(x)**2, x)")
        nb.add_cell(cell)
        result = nb.evaluate_cell(cell.id)
        self.assertEqual(result.status.value, "success")
        self.assertEqual(result.outputs[0].data["text/plain"], "2*sin(x)*cos(x)")

    def test_integral(self):
        nb = Notebook(self.engine)
        cell = Cell.new_code("integrate(x**2, x)")
        nb.add_cell(cell)
        result = nb.evaluate_cell(cell.id)
        self.assertEqual(result.status.value, "success")
        self.assertEqual(result.outputs[0].data["text/plain"], "x**3/3")

    def test_solve_equation(self):
        nb = Notebook(self.engine)
        cell = Cell.new_code("solve(x**2 - 4, x)")
        nb.add_cell(cell)
        result = nb.evaluate_cell(cell.id)
        self.assertEqual(result.status.value, "success")
        self.assertEqual(result.outputs[0].data["text/plain"], "[-2, 2]")

    def test_matrix(self):
        nb = Notebook(self.engine)
        cell = Cell.new_code("Matrix([[1, 2], [3, 4]])")
        nb.add_cell(cell)
        result = nb.evaluate_cell(cell.id)
        self.assertEqual(result.status.value, "success")
        self.assertIn("Matrix", result.outputs[0].data["text/plain"])

    def test_complex_numbers(self):
        nb = Notebook(self.engine)
        cell = Cell.new_code("expand((3 + 4*I)**2)")
        nb.add_cell(cell)
        result = nb.evaluate_cell(cell.id)
        self.assertEqual(result.status.value, "success")
        self.assertEqual(result.outputs[0].data["text/plain"], "-7 + 24*I")

    def test_units(self):
        nb = Notebook(self.engine)
        cell = Cell.new_code("5*meter/second")
        nb.add_cell(cell)
        result = nb.evaluate_cell(cell.id)
        self.assertEqual(result.status.value, "success")
        self.assertEqual(result.outputs[0].data["text/plain"], "5*meter/second")

    def test_unit_conversion(self):
        nb = Notebook(self.engine)
        cell = Cell.new_code("convert(5*meter/second, kilometer/hour)")
        nb.add_cell(cell)
        result = nb.evaluate_cell(cell.id)
        self.assertEqual(result.status.value, "success")
        self.assertEqual(result.outputs[0].data["text/plain"], "18*kilometer/hour")

    def test_persistence(self):
        nb = Notebook(self.engine)
        cell = Cell.new_code("x = 42")
        nb.add_cell(cell)
        nb.evaluate_cell(cell.id)

        filepath = self.temp_path / "test.mathnb"
        nb.save(filepath)

        nb2 = Notebook.load(filepath, self.engine)
        self.assertEqual(len(nb2.cells), 1)
        self.assertEqual(nb2.cell_order[0], cell.id)

        cell2 = Cell.new_code("x")
        nb2.add_cell(cell2)
        result = nb2.evaluate_cell(cell2.id)
        self.assertEqual(result.outputs[0].data["text/plain"], "42")

    def test_cycle_detection(self):
        nb = Notebook(self.engine)
        cell1 = Cell.new_code("x = y + 1")
        cell2 = Cell.new_code("y = x + 1")
        nb.add_cell(cell1)
        nb.add_cell(cell2)

        # The second cell should detect a cycle
        result2 = nb.evaluate_cell(cell2.id)
        # Cycle detection marks the cell as error
        self.assertEqual(result2.status.value, "error")

    def test_evaluate_all(self):
        nb = Notebook(self.engine)
        cell1 = Cell.new_code("a = 1")
        cell2 = Cell.new_code("b = a + 1")
        cell3 = Cell.new_code("c = b + 1")
        nb.add_cell(cell1)
        nb.add_cell(cell2)
        nb.add_cell(cell3)

        results = nb.evaluate_all()
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertEqual(r.status.value, "success")

    def test_rational_arithmetic(self):
        nb = Notebook(self.engine)
        cell = Cell.new_code("Rational(1, 3) + Rational(1, 6)")
        nb.add_cell(cell)
        result = nb.evaluate_cell(cell.id)
        self.assertEqual(result.status.value, "success")
        self.assertEqual(result.outputs[0].data["text/plain"], "1/2")

    def test_latex_output(self):
        nb = Notebook(self.engine)
        cell = Cell.new_code("x**2 + y**2")
        nb.add_cell(cell)
        result = nb.evaluate_cell(cell.id)
        self.assertEqual(result.status.value, "success")
        self.assertIn("text/latex", result.outputs[0].data)
        latex = result.outputs[0].data["text/latex"]
        self.assertIn("x^{2}", latex)
        self.assertIn("y^{2}", latex)


if __name__ == "__main__":
    unittest.main()