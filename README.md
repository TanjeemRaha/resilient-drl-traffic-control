# Resilient DRL Traffic Control

A reproducible Python research project comparing FixedTime, Actuated, DQN, standard PPO, and disruption-trained resilient PPO control at a synthetic four-approach intersection.

> **Research scope:** This is a synthetic proof of concept, not a field-calibrated traffic model. Flooding is represented as reduced road capacity. Results must not be interpreted as real-world intersection performance.

## Skills required

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Reinforcement Learning](https://img.shields.io/badge/Reinforcement_Learning-DRL-7B2CBF)
![Gymnasium](https://img.shields.io/badge/Gymnasium-Environment-008080)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-DQN_%7C_PPO-EF476F)
![Traffic Simulation](https://img.shields.io/badge/Traffic-Simulation-F4A261)
![Data Analysis](https://img.shields.io/badge/Data-Analysis-2A9D8F)
![Testing](https://img.shields.io/badge/Testing-pytest-0A9EDC?logo=pytest&logoColor=white)

## Design

- Seeded, one-second queue simulation with finite storage, fractional service flow, protected north-south/east-west greens, yellow transitions, minimum and maximum green times, and exact vehicle-delay accounting.
- A 12-value continuous observation and two discrete phase-request actions exposed through a Gymnasium environment.
- Conventional FixedTime and Actuated baselines alongside DQN, standard PPO, and resilient PPO.
- Seven configurable evaluation scenarios, including demand peaks, lane closure, flood and incident proxies, an unseen combined disruption, and spillback stress.
- Fair evaluation using shared traffic seeds, deterministic learned actions, conservation checks, paired controller comparisons, hierarchical bootstrap intervals, and model-behavior audits.
- Quick and full profiles with profile-aware model metadata checks, evaluation validation, plotting, reporting, and final audit.

The quick profile is a workflow smoke test: one trained-model seed, two traffic seeds, and 5,000 training steps per learned controller. The full profile uses trained-model seeds 0, 1, and 2; 20 traffic seeds; and 150k/250k/300k steps for DQN, standard PPO, and resilient PPO respectively.

## Setup

Python 3.11 is recommended.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

## Run

Run tests and environment validation:

```powershell
python -m pytest -q
python scripts/validate_environment.py
```

Run the complete quick workflow:

```powershell
python scripts/run_pipeline.py --profile quick
```

Run the complete full experiment:

```powershell
python scripts/run_pipeline.py --profile full
```

Compatible existing models are reused. Add `--force-train` only when intentional retraining is required. Each pipeline ends with the matching profile-aware audit:

```powershell
python scripts/audit_results.py --profile quick
python scripts/audit_results.py --profile full
```

Individual stages are available through `run_baselines.py`, `train_agents.py`, `evaluate_controllers.py`, and `generate_figures.py` in `scripts/`.

## Outputs

```text
configs/                 experiment and scenario configuration
src/resilient_traffic/   simulator, environment, training, evaluation, analysis
scripts/                 validation and pipeline entry points
tests/                   unit and integration tests
models/                  final models and metadata
logs/                    Monitor, evaluation, and TensorBoard data
results/raw/             second-level evaluation records
results/summaries/       episode metrics, comparisons, audits, research summary
results/figures/         publication-style PNG and PDF figures
```

The main report is `results/summaries/research_summary.md`. Quick-profile reports are explicitly labeled smoke-test measurements and do not make controller-superiority claims. Full-profile reports summarize all scenarios, paired differences against FixedTime and Actuated with confidence intervals, and learned-controller behavior agreement.

## Metrics and statistics

Primary measures include mean accumulated wait across accepted vehicles, time-average and maximum queue, throughput, ending queue, overflow, directional imbalance, maximum individual wait, switching, and recovery time. The accounting identities are:

```text
total arrivals = throughput + ending queue + overflow
cumulative delay = departed-vehicle waits + current waits of ending-queue vehicles
```

Learned-controller uncertainty uses a hierarchical bootstrap over trained-model and traffic seeds. Conventional controllers bootstrap traffic seeds only. Paired comparisons preserve shared traffic realizations; a difference is treated as uncertain when its confidence interval includes zero. Multi-seed learning curves retain separate seed histories and align their mean on training timesteps.

## Limitations

The simulator omits real geometry, turning movements, pedestrians, detector noise, platoons, route choice, vehicle dynamics, network spillback, and hardware constraints. Reward weights and disruption distributions are experimental assumptions. Field claims require calibration and validation with observed data or a microscopic simulator such as SUMO or VISSIM.

## License

See [LICENSE](LICENSE).
