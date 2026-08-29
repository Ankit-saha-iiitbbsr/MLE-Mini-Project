# Submission Checklist & Rubric Mapping

**PCAM\* ZC412 · EC-1 Mini-Project · Flavor B — Image-Based Defect / Quality Classifier**

Maps every required deliverable and every rubric criterion to the artifact that
satisfies it.

---

## Deliverables (Brief §6)

### 1 · Versioned dataset and pipeline code

| item | where |
| --- | --- |
| Repository | this repo — commit history reflects weekly progress |
| Data pipeline | [`src/defectvision/data/`](../src/defectvision/data/) |
| Dataset version | `data/processed/manifest.csv` + `dataset_card.json` |
| Pipeline DAG | [`dvc.yaml`](../dvc.yaml) — `dvc dag` to view |
| Config | [`params.yaml`](../params.yaml) — single source of truth |
| Notes | [`docs/DATA.md`](DATA.md) |

### 2 · Experiment tracking logs and model comparison report

| item | where |
| --- | --- |
| Tracking store | `mlflow.db` — `mlflow ui --backend-store-uri sqlite:///mlflow.db` |
| Comparison report | `reports/model_comparison.md` *(generated)* |
| Comparison table | `reports/model_comparison.csv` *(generated)* |
| Per-run artifacts | MLflow → `reproducibility/`, `figures/`, `metrics/` |
| Training curves | `reports/figures/<model>/training_history.png` |

### 3 · Deployed model with a working API endpoint

| item | where |
| --- | --- |
| Service | [`src/defectvision/serving/app.py`](../src/defectvision/serving/app.py) |
| Deployable artifact | `models/production/model_bundle.pt` |
| Sample calls | [`docs/API_EXAMPLES.md`](API_EXAMPLES.md) — curl for every path |
| Postman collection | [`docs/postman_collection.json`](postman_collection.json) |
| Interactive docs | `http://localhost:8000/docs` when running |
| Container | [`Dockerfile`](../Dockerfile), [`docker-compose.yml`](../docker-compose.yml) |
| Latency evidence | `reports/api_benchmark.json` *(generated)* |

### 4 · Monitoring log, drift-simulation report, retraining trigger design

| item | where |
| --- | --- |
| Prediction log | `monitoring/predictions.db`, exported to `reports/prediction_log.jsonl` |
| Drift simulation | `reports/drift_simulation.json` *(generated)* |
| Drift report | `reports/drift_report.md` *(generated)* |
| Monitoring figures | `reports/figures/monitoring/` |
| Trigger design | [`docs/RETRAINING_DESIGN.md`](RETRAINING_DESIGN.md) |
| Live decision | `reports/retraining_decision.json` *(generated)* |

### 5 · README with architecture diagram, setup, and demo

| item | where |
| --- | --- |
| README + diagram | [`README.md`](../README.md) — Mermaid architecture diagram |
| Setup | README → *Quick start* |
| Full execution guide | [`docs/RUNBOOK.md`](RUNBOOK.md) — every stage, expected output, troubleshooting |
| **Design justifications** (Brief §4) | **[`docs/DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md)** — model choice, drift-detection approach and retraining trigger, each with alternatives considered and the measured evidence; README → *Design decisions* is the summary |
| Demo running order | [`docs/DEMO_SCRIPT.md`](DEMO_SCRIPT.md) — timed to 6:00 |

---

## Rubric mapping (Brief §7)

### Data Engineering & Versioning (M2) — 20%

| assessed | evidence |
| --- | --- |
| Ingestion logic | Three interchangeable sources behind one output contract — [`data/acquire.py`](../src/defectvision/data/acquire.py) |
| Validation logic | Per-file quarantine + corpus-level abort gates, split by severity — [`data/validate.py`](../src/defectvision/data/validate.py) |
| Feature engineering | Seven-statistic module shared by training *and* serving, which is the concrete defence against skew — [`features/image_stats.py`](../src/defectvision/features/image_stats.py) |
| Dataset versioning | Manifest + card as DVC-tracked artifacts; manifest SHA-256 logged to every run |

**Beyond the baseline:** near-duplicate detection with grouped splitting (a
leakage class most submissions miss), and a shortcut-leakage check that
quantifies whether any single global statistic separates the classes. Both found
real issues in the real corpus: 64 exact and 412 near-duplicates.

### Experimentation & Reproducibility (M3) — 20%

| assessed | evidence |
| --- | --- |
| Multiple tracked experiments | Four arms — from-scratch CNN, two transfer models, and a classical control |
| Clear comparison | `reports/model_comparison.md` with bootstrap CIs, generated from tracked runs |
| Reproduce from logged config | Git commit + config hash + **dataset manifest hash** + library versions, in `reproducibility/` on every run — [`training/reproducibility.py`](../src/defectvision/training/reproducibility.py) |

**Beyond the baseline:** the classical control makes "we used deep learning" a
measured decision; bootstrap confidence intervals make the comparison honest
about what 1,457 test images can resolve; and promotion is a **gated rule** with
a latency tie-break, not a glance at a table.

### Model Packaging & Deployment (M4) — 20%

| assessed | evidence |
| --- | --- |
| Correct packaging | Self-contained bundle: weights + preprocessing spec + classes + threshold — [`bundle.py`](../src/defectvision/bundle.py) |
| Working REST API | 7 endpoints, OpenAPI docs, Docker image |
| Input validation | Typed 4xx for every malformed-input path, each with a `hint` |
| Latency/throughput awareness | Measured per training run *and* end-to-end over HTTP |

**Beyond the baseline:** preprocessing ships inside the bundle so post-deploy
config edits cannot change how the model sees an image; liveness and readiness
are separate endpoints; batch is partially fault-tolerant; the serving image
deliberately excludes MLflow.

### Monitoring, Drift & Retraining (M5) — 20%

| assessed | evidence |
| --- | --- |
| Prediction logging | SQLite log with decision, image statistics, model identity, delayed ground truth — [`monitoring/store.py`](../src/defectvision/monitoring/store.py) |
| Realistic drift simulation | Physically-motivated corruptions **plus one real, un-simulated shift** |
| Meaningful signals | PSI (frozen bins), KS, chi-square, confidence, review rate, accuracy |
| Sound retraining trigger | Persistence + cooldown + label sufficiency + severity routing |

**Beyond the baseline:** a `baseline` control scenario establishing the noise
floor (so PSI is read against it, not against zero); the `real_camera_upgrade`
scenario built from a capture deliberately held back from training; and a
trigger that can conclude **`monitor`** or **`investigate_capture`** rather than
reflexively retraining.

### Documentation & Presentation — 20%

| assessed | evidence |
| --- | --- |
| README clarity | Architecture diagram, quick start, per-module walkthrough |
| Architecture diagram | Mermaid, showing the full loop including feedback |
| Code organisation | Package mirrors the course modules; 122 tests; CI |
| Demo quality | [`docs/DEMO_SCRIPT.md`](DEMO_SCRIPT.md), timed with anticipated questions |

**Beyond the baseline:** every non-obvious decision is justified where it lives
(module docstrings) *and* summarised in the README, with the reasoning — not
just the choice.

---

## Reproducing every artifact

```bash
python -m defectvision.cli data && python -m defectvision.cli train --all && python -m defectvision.cli compare
```

```bash
python -m defectvision.cli reference-stats && python -m defectvision.cli simulate-drift && python -m defectvision.cli monitor && python -m defectvision.cli check-retrain
```

Or, honouring the DAG and skipping stages that are already current:

```bash
dvc repro
```

---

## Pre-submission checks

```bash
pytest -q
```

```bash
ruff check src tests
```

```bash
git log --oneline
```

- [ ] Tests pass
- [ ] `reports/` regenerated from a clean run
- [ ] `git status` clean; no `kaggle.json`, `.env`, or dataset files committed
- [ ] README architecture diagram renders on the hosting platform
- [ ] Demo recorded (5–7 min) following `docs/DEMO_SCRIPT.md`
- [ ] Repository link shared with the required access
