# Feature, model and evaluation pipeline

```
models/
  features/build_features.py     run logs + data/indices/development.csv -> results/features/
  meldloc/meldloc_core.py        the model (threshold augmentation, core + adapter, calibration, bound)
  meldloc/train.py               training with the fixed seed -> models/frozen/
  meldloc/*.py                   scoring and analyses (table below)
  baselines/*.py                 methods compared against, not part of the model
  frozen/                        the three trained axis models used for every result in results/
results/<analysis>/              one folder per analysis
reports/                         summary tables written by report_tables.py
```

| stage | script | reads | writes |
|---|---|---|---|
| features and labels | `features/build_features.py` | `data/indices/development.csv`, run logs | `results/features/development_features.npz`, `cache_manifest.json` |
| training (seed 20260803) | `meldloc/train.py` | features | `models/trained/meldloc_{lon,lat,yaw}.joblib`, `models/trained/train_report.json` |
| development-validation scoring | `meldloc/devval_scoring.py` | features, trained models | `results/development/devval_scoring.json` |
| distributional baselines (Q-multi, K-indep) | `baselines/distributional_baselines.py` | features | `results/baselines/distributional_baselines.json` |
| reference methods (fixed 3-sigma, matcher score, B0) | `baselines/reference_methods.py` | features | `results/baselines/reference_methods.json` |
| survival stacking, rearranged Q-multi, feature-group pruning | `meldloc/baselines_pruning.py` | features | `results/baselines/{survival_stacking,rearranged_qmulti}.json`, `results/ablations/feature_groups.json` |
| risk-coverage curve | `meldloc/risk_coverage.py` | features, trained models | `results/development/risk_coverage.json` |
| bound tightness | `meldloc/bound_tightness.py` | features, trained models | `results/development/bound_tightness.json` |
| model-component ablations | `meldloc/ablations.py` | features | `results/ablations/model_components.json` |
| holdout / cross-route (one shot) | `meldloc/holdout_eval.py`, `meldloc/crossroute_eval.py` | run logs + ledgers, trained models | `results/features/{holdout,crossroute}_features.npz`, `results/evaluation/*.json` |
| seed variability | `meldloc/seed_variability.py`, `meldloc/seed_variability_summary.py` | features | `results/seeds/` |
| conditional coverage, cap vs raw, alarm delay, latency | `meldloc/coverage_timing_analyses.py` | features, trained models (+ run logs for the alarm delay) | `results/coverage/`, `results/timing/` |
| stress sets and the matcher's Hessian covariance | `meldloc/stress_eval.py` | run logs + ledgers of the stress and ndt_covariance roles, trained models | `results/stress/` |
| summary tables | `meldloc/report_tables.py` | all of the above | `reports/*.tex`, `reports/tables.md` |

Quick check from the shipped files (no CARLA, no run logs):

```
pip install -r requirements.txt
python3 models/meldloc/report_tables.py
python3 models/meldloc/devval_scoring.py     # re-scores the development-validation split (re-fits the baselines, ~2 min)
python3 models/meldloc/seed_variability_summary.py
```

`models/frozen/` holds the models every result file was computed with; `train.py` writes to `models/trained/`
because a retraining reproduces the metrics closely but not bit-identically (LightGBM's histogram layout and
thread scheduling are not deterministic), and the evaluation scripts verify the shipped models against
`results/evaluation/model_freeze_manifest.json` before scoring.

`build_features.py` reproduces the frozen cache (237 runs; the 18 development runs marked `frozen_cache = 0`
in the index are skipped unless `--all` is given, see `data/indices/README.md`).

Scripts marked "run logs" need the run directories under `data/logs/<role>/` and the per-role
`ledger.csv` (both in the data archive, or produced by `datagen/generate_<role>.sh`); the shipped
`results/features/*.npz` files are their outputs, so the later stages run without the logs.
Scripts that use the trained models first re-verify `results/evaluation/model_freeze_manifest.json`.
