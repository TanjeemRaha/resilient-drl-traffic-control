# Resilient DRL Traffic Control

Traffic-signal control experiments for a synthetic four-way intersection. The project compares FixedTime, Actuated, DQN, PPO, and a disruption-trained PPO controller under normal traffic and six disruption scenarios.

This is a proof of concept built on a queue-based simulator. It is not calibrated with field data and should not be used to make claims about a real intersection.

## Requirements

- Python 3.11
- A working C/C++ runtime for PyTorch
- Enough disk space for trained models and experiment outputs

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

## Run

Use the quick profile to check the complete workflow:

```powershell
python scripts/run_pipeline.py --profile quick
```

Run the full experiment when the quick pipeline passes:

```powershell
python scripts/run_pipeline.py --profile full
```

The pipeline runs the tests, validates the environment, evaluates the baselines, trains any missing models, evaluates all controllers, creates figures, and runs the profile-specific audit. Compatible models are reused unless `--force-train` is supplied.

Run individual checks with:

```powershell
python -m pytest -q
python scripts/validate_environment.py
python scripts/audit_results.py --profile quick
python scripts/audit_results.py --profile full
```

## Experiment profiles

| Profile | Model seeds | Traffic seeds | Training steps | Raw output |
| --- | ---: | ---: | --- | --- |
| `quick` | 1 | 2 | 5,000 per controller | Complete second-level records |
| `full` | 3 | 20 | DQN: 150k, PPO: 250k, resilient PPO: 300k | Episode results plus compact audit files |

Profile settings are in `configs/quick.yaml` and `configs/full.yaml`. Traffic scenarios are defined in `configs/scenarios.yaml`.

## Results

- `results/summaries/episode_metrics.csv`: one row per evaluation episode
- `results/summaries/controller_summary.csv`: controller results and confidence intervals
- `results/summaries/paired_controller_comparisons.csv`: paired comparisons against the baselines
- `results/summaries/research_summary.md`: generated experiment report
- `results/raw/decision_actions.csv`: one row per learned-controller decision
- `results/raw/example_time_series.csv`: second-level data for the configured example
- `results/figures/`: PNG and PDF plots

Full-profile second-level records are not committed because of their size. The configurations, seeds, model metadata, episode summaries, comparison tables, compact audit files, and figures are retained so the experiment can be reviewed and reproduced.

## License

MIT. See [LICENSE](LICENSE).
