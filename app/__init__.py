"""
Production layer for the event-extraction pipeline.

The historical training code lives in ``src/dateclf`` (not packaged).
This shim makes it importable from the API/tests/Docker without
requiring an editable install.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
