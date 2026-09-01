from __future__ import annotations

import json

import pandas as pd
import pytest

from resilient_traffic.auditing import ALL_CONTROLLERS, BASELINES, validate_model_metadata, validate_profile_coverage
from resilient_traffic.config import load_config
from resilient_traffic.training import AGENTS


def _episode_design(config, learned_seeds=None, profile=None):
    learned_seeds = list(config["training"]["seeds"] if learned_seeds is None else learned_seeds)
    rows = []
    for controller in ALL_CONTROLLERS:
        model_seeds = [-1] if controller in BASELINES else learned_seeds
        for model_seed in model_seeds:
            for scenario in config["scenarios"]:
                for traffic_seed in config["evaluation"]["traffic_seeds"]:
                    for episode in range(config["evaluation"]["episodes_per_seed"]):
                        rows.append({"profile": profile or config["profile"],
                                     "output_label": config["label"], "controller": controller,
                                     "trained_model_seed": model_seed, "scenario": scenario,
                                     "traffic_seed": traffic_seed, "episode": episode})
    return pd.DataFrame(rows)


def test_quick_audit_rejects_full_profile_output():
    quick, full = load_config("quick"), load_config("full")
    with pytest.raises(ValueError, match="profile mismatch"):
        validate_profile_coverage(_episode_design(full), quick)


def test_full_audit_rejects_missing_model_seeds():
    full = load_config("full")
    with pytest.raises(ValueError, match="model-seed mismatch"):
        validate_profile_coverage(_episode_design(full, learned_seeds=[0]), full)


def test_full_audit_checks_every_model_metadata_file(tmp_path):
    full = load_config("full")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    for controller, specification in AGENTS.items():
        for seed in full["training"]["seeds"]:
            stem = model_dir / f"{controller}_seed{seed}"
            stem.with_suffix(".zip").write_bytes(b"model")
            metadata = {"profile": "full", "timesteps": full["training"]["timesteps"][controller],
                        "reward_configuration": full["reward"], "algorithm": specification["algorithm"],
                        "training_mode": specification["mode"], "reward": specification["reward"]}
            stem.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    (model_dir / "PPO_resilient_seed2.json").unlink()
    with pytest.raises(FileNotFoundError, match="PPO_resilient.*seed 2"):
        validate_model_metadata(full, tmp_path)
