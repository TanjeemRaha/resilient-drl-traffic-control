"""Stable-Baselines3 DQN/PPO training orchestration."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from .config import ROOT
from .environment import TrafficSignalEnv
from .utils import ensure_output_dirs, metadata, save_json, set_global_seed

AGENTS: dict[str, dict[str, str]] = {
    "DQN_standard": {"algorithm": "DQN", "mode": "normal_training", "reward": "queue_reward"},
    "PPO_standard": {"algorithm": "PPO", "mode": "normal_training", "reward": "queue_reward"},
    "PPO_resilient": {"algorithm": "PPO", "mode": "resilient_training", "reward": "resilience_reward"},
}


def model_path(name: str, seed: int) -> Path:
    return ROOT / "models" / f"{name}_seed{seed}.zip"


def train_agents(config: dict[str, Any], force: bool = False) -> list[Path]:
    """Train the three specified agents, skipping existing final models by default."""
    ensure_output_dirs()
    outputs: list[Path] = []
    for name, spec in AGENTS.items():
        for seed in config["training"]["seeds"]:
            destination = model_path(name, seed)
            outputs.append(destination)
            if destination.exists() and not force:
                metadata_path = destination.with_suffix(".json")
                existing = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
                desired_steps = int(config["training"]["timesteps"][name])
                if (existing.get("profile") == config["profile"]
                        and existing.get("timesteps") == desired_steps
                        and existing.get("reward_configuration") == config["reward"]):
                    print(f"Skipping compatible existing model: {destination.name}")
                    continue
                print(f"Existing {destination.name} belongs to another profile/timestep setting; retraining.")
            set_global_seed(seed)
            log_dir = ROOT / "logs" / f"{name}_seed{seed}"
            log_dir.mkdir(parents=True, exist_ok=True)
            train_env = Monitor(TrafficSignalEnv(config, reward_name=spec["reward"],
                                                  training_mode=spec["mode"]),
                                str(log_dir / "monitor.csv"))
            eval_env = Monitor(TrafficSignalEnv(config, reward_name=spec["reward"],
                                                 training_mode=spec["mode"]))
            checkpoint = CheckpointCallback(save_freq=int(config["training"]["checkpoint_frequency"]),
                                            save_path=str(ROOT / "models" / "checkpoints"),
                                            name_prefix=f"{name}_seed{seed}")
            evaluation = EvalCallback(eval_env, eval_freq=int(config["training"]["eval_frequency"]),
                                      n_eval_episodes=int(config["training"]["eval_episodes"]),
                                      deterministic=True, best_model_save_path=str(log_dir / "best"),
                                      log_path=str(log_dir / "eval"))
            common = dict(policy="MlpPolicy", env=train_env, seed=seed, device="cpu", verbose=0,
                          tensorboard_log=str(log_dir / "tensorboard"), policy_kwargs={"net_arch": [64, 64]})
            if spec["algorithm"] == "DQN":
                hyperparameters: dict[str, Any] = dict(learning_rate=1e-3, buffer_size=10_000,
                    learning_starts=500, batch_size=64, gamma=0.99, train_freq=4,
                    target_update_interval=500, exploration_fraction=0.25,
                    exploration_final_eps=0.05)
                model = DQN(**common, **hyperparameters)
            else:
                hyperparameters = dict(learning_rate=3e-4, n_steps=256, batch_size=64,
                                       n_epochs=5, gamma=0.99, gae_lambda=0.95,
                                       ent_coef=0.01, clip_range=0.2)
                model = PPO(**common, **hyperparameters)
            timesteps = int(config["training"]["timesteps"][name])
            print(f"Training {name}, seed={seed}, timesteps={timesteps}")
            model.learn(total_timesteps=timesteps, callback=[checkpoint, evaluation],
                        tb_log_name="run", progress_bar=False)
            model.save(destination.with_suffix(""))
            save_json(destination.with_suffix(".json"), metadata(seed, controller=name,
                algorithm=spec["algorithm"], training_mode=spec["mode"], reward=spec["reward"],
                reward_configuration=config["reward"],
                timesteps=timesteps, profile=config["profile"], output_label=config["label"],
                policy_network=[64, 64], hyperparameters=hyperparameters))
            train_env.close()
            eval_env.close()
    return outputs


def load_model(name: str, seed: int):
    """Load a trained controller with a clear missing-file error."""
    path = model_path(name, seed)
    if not path.exists():
        raise FileNotFoundError(f"Required model is missing: {path}. Run scripts/train_agents.py first.")
    return DQN.load(path) if AGENTS[name]["algorithm"] == "DQN" else PPO.load(path)
