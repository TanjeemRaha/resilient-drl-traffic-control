"""Filesystem, reproducibility, and metadata helpers."""
from __future__ import annotations

import json
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
import pandas as pd
import stable_baselines3
import torch

from .config import ROOT


def ensure_output_dirs() -> None:
    for relative in ("models", "models/checkpoints", "logs", "results/raw", "results/summaries", "results/figures"):
        (ROOT / relative).mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def package_versions() -> dict[str, str]:
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "torch": torch.__version__, "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__}


def metadata(seed: int, **values: Any) -> dict[str, Any]:
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
            "versions": package_versions(), **values}

