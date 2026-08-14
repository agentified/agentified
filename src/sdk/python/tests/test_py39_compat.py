"""Guard the declared Python 3.9 runtime floor (`requires-python >=3.9`).

`from __future__ import annotations` defers annotations only; module- and
class-level *assignments* (typing aliases like ``X = Callable[..., A | B]``)
still evaluate at import time, and PEP 604 ``|`` unions raise ``TypeError``
on CPython 3.9. This scan fails on any such alias so the package stays
importable on every version the wheel claims to support (ADR-0006 ships
abi3 wheels for >=3.9).
"""

import ast
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent / "ratel_ai"


def test_no_runtime_evaluated_pep604_unions_at_import_time() -> None:
    offenders: list[str] = []
    for module_path in sorted(PACKAGE_DIR.rglob("*.py")):
        tree = ast.parse(module_path.read_text(), filename=str(module_path))
        for assignment in _import_time_assignments(tree):
            for union in _pep604_unions(assignment):
                offenders.append(
                    f"{module_path.relative_to(PACKAGE_DIR)}:{union.lineno}: "
                    f"{ast.unparse(union)}"
                )
    assert not offenders, (
        "PEP 604 unions evaluated at import time break CPython 3.9 "
        "(use typing.Optional/typing.Union in runtime-evaluated aliases):\n"
        + "\n".join(offenders)
    )


def _import_time_assignments(tree: ast.Module) -> list[ast.expr]:
    """Values of module- and class-level assignments — evaluated at import."""
    values: list[ast.expr] = []
    class_bodies = [
        statement for node in tree.body if isinstance(node, ast.ClassDef) for statement in node.body
    ]
    for statement in [*tree.body, *class_bodies]:
        if isinstance(statement, ast.Assign):
            values.append(statement.value)
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            values.append(statement.value)
    return values


def _pep604_unions(value: ast.expr) -> list[ast.BinOp]:
    """``a | b`` nodes that are type unions rather than integer/set bit-ors."""
    return [
        node
        for node in ast.walk(value)
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.BitOr)
        and any(_is_type_expression(operand) for operand in (node.left, node.right))
    ]


def _is_type_expression(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant) and node.value is None:
        return True
    return isinstance(node, ast.Subscript)
