"""Test package bootstrap.

Puts the src layout on sys.path so every test module imports `science2code`
regardless of how the suite is invoked: `unittest discover`, a single module
via `python -m unittest tests.test_server`, pytest, or an editor's runner.

Why this exists. Four of the six test modules did their own sys.path insert and
two did not. Discovery still passed, because `test_anchor` sorts first and its
insert happened to make the package importable for everything after it. That is
a real fragility rather than a cosmetic one: running one module on its own
failed, and the suite's success depended on alphabetical ordering. Bootstrapping
once here removes the ordering dependency entirely.

An installed package (`pip install -e .`) takes precedence; this only helps when
the package is not installed.
"""

import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
