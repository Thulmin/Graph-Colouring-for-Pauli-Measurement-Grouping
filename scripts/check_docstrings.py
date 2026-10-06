"""Requirement R18: every public function, class and method in the package has a docstring.

Walks src/dtp with the ast module (no import side effects). "Public" means the name does not
start with an underscore; methods of private classes and nested functions are skipped.
Exit status 0 if all public objects are documented, 1 otherwise.

Usage: python scripts/check_docstrings.py
Original code written for this project.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "dtp"


def missing_docstrings() -> tuple[list[str], int]:
    missing: list[str] = []
    total = 0
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(SRC.parent)
        total += 1
        if not ast.get_docstring(tree):
            missing.append(f"{rel}: module docstring")
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
                total += 1
                if not ast.get_docstring(node):
                    missing.append(f"{rel}:{node.lineno} {node.name}")
                if isinstance(node, ast.ClassDef):
                    for sub in node.body:
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and not sub.name.startswith("_"):
                            total += 1
                            if not ast.get_docstring(sub):
                                missing.append(f"{rel}:{sub.lineno} {node.name}.{sub.name}")
    return missing, total


def main() -> int:
    missing, total = missing_docstrings()
    print(f"public objects checked: {total}; missing docstrings: {len(missing)}")
    for item in missing:
        print("  " + item)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
