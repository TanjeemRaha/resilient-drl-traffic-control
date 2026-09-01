"""Resilient traffic-control proof-of-concept package."""
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "logs" / ".matplotlib"))

__version__ = "0.1.0"
