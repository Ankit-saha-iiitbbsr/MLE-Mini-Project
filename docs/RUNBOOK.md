# Runbook — Complete Step-by-Step Execution Guide

**PCAM\* ZC412 · EC-1 Mini-Project · Flavor B — Image-Based Defect / Quality Classifier**

Every command needed to run this project from a fresh clone, what each stage
does, what it produces, and how to confirm it worked.

**Total runtime on CPU: ~40 minutes** (dominated by training). A 3-minute smoke
path is in [§9](#9-fast-smoke-run-3-minutes).

---

## Contents

| § | Stage | Command | Time |
| --- | --- | --- | --- |
| [0](#0-prerequisites) | Prerequisites | — | — |
| [1](#1-environment-setup) | Environment setup | `python -m venv .venv` … | ~4 min |
| [2](#2-verify-the-setup) | Verify setup | `defectvision info` | 5 s |
| [3](#3-m2--data-pipeline) | **M2** Data pipeline | `defectvision data` | ~3 min |
| [4](#4-m3--train-and-track-experiments) | **M3** Train | `defectvision train --all` | ~31 min |
| [5](#5-m3--compare-gate-and-promote) | **M3** Compare & promote | `defectvision compare` | 5 s |
| [6](#6-m4--serve-the-model) | **M4** Serve | `defectvision serve` | continuous |
| [7](#7-m4--test-and-benchmark-the-api) | **M4** Test & benchmark | `curl`, `defectvision benchmark` | ~1 min |
| [8](#8-m5--monitoring-drift-and-retraining) | **M5** Monitoring | `reference-stats` → `check-retrain` | ~7 min |
| [9](#9-fast-smoke-run-3-minutes) | Fast smoke run | with overrides | ~3 min |
| [10](#10-reproducing-a-specific-run) | Reproduce a run | `dvc repro` | varies |
| [11](#11-troubleshooting) | Troubleshooting | — | — |

---

## 0. Prerequisites

| requirement | notes |
| --- | --- |
| **Python 3.10+** | Developed on 3.12.10. No GPU needed. |
| **~4 GB free disk** | Dependencies ~2.5 GB, dataset ~100 MB, artifacts ~50 MB. |
| **Git** | For cloning and for the reproducibility metadata each run logs. |
| Docker *(optional)* | Only for §6b. The project runs fully without it. |
| Kaggle account *(optional)* | Only if you do **not** already have the dataset. See §3a. |

> **On this machine:** bare `python` hits the Microsoft Store alias and fails.
> Use the `py` launcher to create the venv; after that, use the venv's own
> interpreter. Commands below assume the venv is activated.

---

## 1. Environment setup

```bash
cd "D:\MLE Project\ML-Engg-mini-project"
```

Create the virtual environment (use `py` on Windows, `python3` on Linux/macOS):

```bash
py -m venv .venv
```

Activate it — **PowerShell**:

```bash
.venv\Scripts\Activate.ps1
```

or **cmd.exe**:

```bash
.venv\Scripts\activate.bat
```

or **bash / Linux / macOS**:

```bash
source .venv/bin/activate
```

Install dependencies (~4 min; PyTorch is the bulk of it):

```bash
python -m pip install --upgrade pip setuptools wheel
```

```bash
pip install -r requirements-dev.txt
```

Install the project itself in editable mode, which also puts the `defectvision`
command on your PATH:

```bash
pip install -e . --no-deps
```

> `make setup` does all of the above in one step if you have `make`.

---

## 2. Verify the setup

```bash
defectvision info
```

You should see JSON reporting the project root, data source, git state,
resolved library versions, and whether a production model exists. On a fresh
clone `production_bundle.exists` will be `false` — that is expected until §5.

Run the test suite to confirm the install is sound (~90 s, 122 tests):

```bash
pytest -q
```

> Every command below can also be run as `python -m defectvision.cli <command>`
> if the `defectvision` shim is not on your PATH.

---

## 3. M2 — Data pipeline

### 3a. Where the data comes from

`data.source` in `params.yaml` decides. It currently reads `kaggle`.

| your situation | what to do |
| --- | --- |
| **Dataset already extracted at `datasets/`** *(the case in this repo)* | Nothing. The stage finds the local copy and skips the network entirely. |
| **No dataset, have a Kaggle token** | Create a token at <https://www.kaggle.com/settings> → *API* → *Create New Token*, move the downloaded `kaggle.json` to `%USERPROFILE%\.kaggle\kaggle.json` (Windows) or `~/.kaggle/kaggle.json` (Linux/macOS, then `chmod 600`). Accept the dataset rules once on its Kaggle page. |
| **No dataset, no Kaggle account** | Set `data.source: synthetic` in `params.yaml`. The full pipeline runs against a procedural generator instead. |

> Kaggle authenticates with an **API token**, never an account password. This
> project never asks for one.

Expected local layout when using the real corpus:

```
datasets/
├── casting_data/casting_data/
│   ├── train/{ok_front, def_front}
│   └── test/{ok_front, def_front}      7348 images @ 300x300
└── casting_512x512/casting_512x512/    1300 images — deliberately NOT trained on
    └── {ok_front, def_front}              (held back as a real drift set for M5)
```

### 3b. Run the data pipeline

```bash
defectvision data
```

This runs three stages in order:

1. **acquire** — normalises the corpus into `data/raw/<class>/` and writes a
   provenance record.
2. **validate** — per-file checks (quarantine bad frames) plus corpus-level
   gates (abort on breach).
3. **split** — stratified, leakage-safe train/val/test manifest.

### 3c. What you should see

```
Kaggle corpus ready: {'ok': 3137, 'defect': 4211} (7348 images)
[PASS] corrupt_file_ratio       0/7348 files quarantined (0.00%); limit 2.00%
[PASS] min_images_per_class     all classes have >= 100 valid images
[PASS] class_imbalance          majority/minority = 1.34 (limit 5.00)
[PASS] exact_duplicate_ratio    64 exact duplicates (0.87%); limit 10.00%
[PASS] near_duplicate_ratio     412 near-duplicates (5.61%) by perceptual hash
[PASS] global_statistic_leakage no single global statistic separates the classes
[PASS] resolution_consistency   1 distinct resolution(s) present
Dropped 64 exact duplicate(s) before splitting
Leakage check passed: 6872 groups, none spanning folds
M2 complete: 7284 images in the versioned manifest
```

Those duplicate counts are real findings in the Kaggle corpus, not warnings you
should ignore — see [`DATA.md`](DATA.md) §3 for how they are handled.

### 3d. Artifacts produced

| file | what it is |
| --- | --- |
| `data/raw/{ok,defect}/` | Normalised corpus |
| `data/raw/_source.json` | Acquisition provenance + citation |
| `data/interim/scan.csv` | Per-file table: hashes, dimensions, 7 image statistics |
| `data/processed/manifest.csv` | **The dataset version** — which image is in which fold |
| `data/processed/dataset_card.json` | Counts, split policy, provenance |
| `reports/validation_report.json` | All gate results |
| `reports/figures/raw_samples.png` | Contact sheet — eyeball this |

### 3e. Optional: exploratory analysis

```bash
python notebooks/01_exploratory_data_analysis.py
```

Prints class balance, the leakage check with effect sizes, and duplicate
counts; writes figures to `reports/figures/eda/`.

---

## 4. M3 — Train and track experiments

```bash
defectvision train --all
```

Trains four arms sequentially, each as a tracked MLflow run. **~31 minutes on
CPU** — measured breakdown:

| arm | what it is | time |
| --- | --- | --- |
| `baseline_cnn` | CNN from scratch, 14 epochs | ~10 min |
| `resnet18` | ImageNet transfer, 6 epochs | ~15 min |
| `mobilenet_v3` | ImageNet transfer, 6 epochs | ~5 min |
| `logreg_hog` | HOG + logistic regression (classical control) | ~35 s |

Train a single arm instead:

```bash
defectvision train --model baseline_cnn
```

### 4a. What you should see

Per epoch:

```
epoch 12/14 | train_loss=0.0422 val_loss=0.0087 | val_f1=0.9992 val_recall=0.9984 val_auc=1.0000 | 41.4s
```

Then per arm:

```
Best epoch: 12 (val f1 = 0.9992)
Operating threshold 0.1600 -- Maximises validation F1 (0.9992) over 501 candidates
TEST  f1=0.9970 recall=0.9952 precision=0.9988 acc=0.9966 auc=1.0000
LATENCY p50=2.0ms p95=2.3ms | batched 621 img/s
```

Note the ordering: the model is selected on **validation**, the threshold is
tuned on **validation**, and test is evaluated once at the end.

### 4b. Inspect the experiments

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open <http://localhost:5000>. Per run you will find:

- **Parameters** — full resolved config
- **Metrics** — per-epoch curves plus final test metrics and latency
- **Artifacts → `reproducibility/`** — git commit, `params.yaml` snapshot,
  dataset manifest SHA-256, library versions, and a `how_to_reproduce.txt`
- **Artifacts → `figures/`** — confusion matrix, ROC, PR, score distribution
- **Artifacts → `model_bundle/`** — the deployable artifact

### 4c. Artifacts produced

| file | what it is |
| --- | --- |
| `mlflow.db` | Tracking store (SQLite) |
| `mlartifacts/` | Per-run artifacts |
| `models/candidates/<model>/model_bundle.pt` | One deployable bundle per arm |
| `reports/figures/<model>/` | Evaluation plots |

---

## 5. M3 — Compare, gate, and promote

```bash
defectvision compare
```

Applies the promotion gates, ranks the survivors, writes the comparison report,
copies the winner to `models/production/`, and registers it in the MLflow Model
Registry.

### 5a. What you should see

```
       model               arch  params_M  test_f1  test_recall  latency_p95_ms
baseline_cnn       baseline_cnn     0.251   0.9970       0.9952            2.27
    resnet18           resnet18    11.171   0.9964       0.9941           19.19
mobilenet_v3 mobilenet_v3_small     0.928   0.9923       0.9893            8.54
  logreg_hog         logreg_hog     0.002   0.9609       0.9620            3.83

gate baseline_cnn   eligible
gate resnet18       eligible
gate mobilenet_v3   eligible
gate logreg_hog     eligible
Selected 'baseline_cnn': Highest test_f1 (0.9970) among candidates clearing the
  gates; it is also the cheapest of the 2 candidates statistically tied with it
Promoted baseline_cnn -> models/production/model_bundle.pt
```

The selection is a **rule**, not a judgement call: gates first, then rank, then
tie-break on latency when scores fall inside the leader's bootstrap confidence
interval.

Promote a specific arm instead (overrides the gated winner):

```bash
defectvision package --model resnet18
```

### 5b. Artifacts produced

| file | what it is |
| --- | --- |
| `models/production/model_bundle.pt` | **The deployed model** — weights + preprocessing + threshold |
| `models/production/PROMOTED.json` | Which model, from which run, and why |
| `reports/model_comparison.md` | The comparison report (a graded deliverable) |
| `reports/model_comparison.csv` | Machine-readable table |

---

## 6. M4 — Serve the model

Leave this running and open a **second terminal** for §7 and §8.

```bash
defectvision serve
```

Ready when you see `Uvicorn running on http://0.0.0.0:8000`.

Interactive OpenAPI docs: <http://localhost:8000/docs>

### 6b. Optional: run it in Docker

```bash
docker build -t defectvision:1.0.0 .
```

```bash
docker run --rm -p 8000:8000 -v "$(pwd)/models/production:/app/models/production:ro" -v "$(pwd)/monitoring:/app/monitoring" defectvision:1.0.0
```

The model is mounted read-only rather than baked in, so promoting a new model
means restarting the container, not rebuilding the image.

---

## 7. M4 — Test and benchmark the API

**In your second terminal** (activate the venv there too).

### 7a. Health and readiness

```bash
curl http://localhost:8000/healthz
```

```bash
curl http://localhost:8000/readyz
```

`/healthz` says the process is alive; `/readyz` says a model is loaded and it
can serve. They are separate on purpose — with no model, `/healthz` still
returns 200 while `/readyz` returns **503**.

### 7b. Model card

```bash
curl http://localhost:8000/model
```

Returns the threshold, the git commit, and the dataset manifest hash — so a
running container traces back to the run that produced it.

### 7c. A real prediction

```bash
curl -F "file=@data/raw/defect/test_cast_def_0_1134.jpeg" http://localhost:8000/predict
```

```json
{
  "predicted_class": "defect",
  "probability_defect": 0.9994,
  "confidence": 0.9994,
  "threshold": 0.16,
  "decision": "auto_reject",
  "latency_ms": 19.0,
  "image_stats": { "mean_intensity": 0.53, "...": "..." }
}
```

And a good part:

```bash
curl -F "file=@data/raw/ok/test_cast_ok_0_1020.jpeg" http://localhost:8000/predict
```

### 7d. Error handling — worth demonstrating

```bash
echo "not an image" > bad.png && curl -w "\nHTTP %{http_code}\n" -F "file=@bad.png" http://localhost:8000/predict
```

Returns **400** with an actionable `hint`, not a 500. Full matrix of error paths
in [`API_EXAMPLES.md`](API_EXAMPLES.md) §5.

### 7e. Load benchmark

```bash
defectvision benchmark --n 200 --concurrency 4
```

Measured end-to-end over HTTP (multipart parsing + decode + inference +
serialisation), not `model.forward()`:

```
Throughput 41.7 req/s | p50=95.8ms p95=107.0ms p99=110.2ms | 200/200 OK
```

Written to `reports/api_benchmark.json`.

### 7f. Postman

Import [`docs/postman_collection.json`](postman_collection.json), attach an
image to the `file` field on the prediction requests, and run the collection.
It exercises the happy paths and the error paths, with assertions.

---

## 8. M5 — Monitoring, drift, and retraining

Run these in the **second terminal**. The API from §6 can keep running — its
predictions are logged to the same store.

### 8a. Build the drift baseline

```bash
defectvision reference-stats
```

Freezes the **training** distribution (samples + PSI bin edges + the model's
behavioural baseline) as the reference. Must be the training distribution: the
question monitoring answers is "does the model still see the kind of data it
learned from".

### 8b. Simulate distribution shift

```bash
defectvision simulate-drift
```

~5 minutes. Pushes eight scenarios through the production model and logs every
prediction:

- `baseline` — uncorrupted test images, the **control** that establishes the
  noise floor
- Five physically-motivated corruptions: `lighting_dim`, `lighting_bright`,
  `camera_angle`, `focus_blur`, `sensor_noise`, `new_variant`
- `real_camera_upgrade` — the held-back `casting_512x512` capture, a **genuine**
  camera shift no corruption operator was tuned against

Run one scenario only:

```bash
defectvision simulate-drift --scenario lighting_dim
```

### 8c. Compute the monitoring signals

```bash
defectvision monitor
```

### 8d. What you should see

```
scenario                 PSI max  #drifted     acc  acc drop    conf  alerts
baseline                   0.051         0  0.9950   +0.0016   0.989  -
focus_blur                12.434         4  0.9925   +0.0041   0.959  data drift
camera_angle              12.434         7  0.9400   +0.0566   0.963  data drift, accuracy degradation
new_variant               12.617         7  0.8375   +0.1591   0.927  data drift, accuracy degradation
real_camera_upgrade        7.344         6  0.7800   +0.2166   0.972  data drift, accuracy degradation
sensor_noise              12.434         6  0.7425   +0.2541   0.910  data drift, confidence collapse, accuracy degradation
lighting_bright           12.617         7  0.6100   +0.3866   0.970  data drift, accuracy degradation
lighting_dim              15.185         7  0.5775   +0.4191   0.996  data drift, accuracy degradation
```

**Read the `lighting_dim` row carefully.** Accuracy collapses from 0.995 to
0.578 while mean confidence *rises* to 0.996 — the model is confidently,
catastrophically wrong. Confidence monitoring alone would have caught 1 of these
7 degradations; PSI caught all 7. That asymmetry is the whole argument for
monitoring the input distribution.

The `baseline` control at PSI 0.051 is what makes the rest readable: it is the
noise floor, so every other row is measured against a known reference rather
than against zero.

### 8e. Evaluate the retraining trigger

```bash
defectvision check-retrain
```

```
should_retrain : False
action         : investigate_capture
reason         : Severe input drift (PSI 6.920). A shift this large usually means a
                 hardware or configuration change (lamp, lens, camera pose) rather
                 than a genuine change in the parts. Inspect the capture rig before
                 retraining -- a model retrained on a broken camera bakes the fault in.
```

Exits **10** when a retrain is warranted, **0** otherwise — so cron or CI can
branch on it. Design rationale in [`RETRAINING_DESIGN.md`](RETRAINING_DESIGN.md).

### 8f. Artifacts produced

| file | what it is |
| --- | --- |
| `monitoring/predictions.db` | The prediction log (SQLite) |
| `monitoring/reference_stats.json` | Frozen drift baseline |
| `reports/drift_report.md` | **Drift analysis** (a graded deliverable) |
| `reports/drift_simulation.json` | Per-scenario metrics |
| `reports/monitoring_report.json` | Per-scenario and per-window signals |
| `reports/retraining_decision.json` | The trigger decision with rule evaluations |
| `reports/prediction_log.jsonl` | Exported monitoring log |
| `reports/figures/monitoring/psi_heatmap.png` | **Look at this one** — the pattern of drifted features names the fault |
| `reports/figures/drift/*.png` | Before/after strips per scenario |

### 8g. Closing the feedback loop

Labels on a real line arrive after the prediction. Attach one:

```bash
curl -X POST http://localhost:8000/feedback -H "Content-Type: application/json" -d "{\"request_id\":\"<id from /predict>\",\"ground_truth\":1,\"source\":\"teardown\"}"
```

---

## 9. Fast smoke run (3 minutes)

Exercises every stage on the synthetic generator at reduced size. Use this to
verify a fresh clone, or in CI.

```bash
defectvision data --source synthetic -s data.synthetic.n_images=240 -s data.image_size=64 -s preprocess.resize=64 -s validation.min_images_per_class=40
```

```bash
defectvision train --model baseline_cnn --no-compare -s train.models.baseline_cnn.epochs=2 -s train.models.baseline_cnn.channels="[8,16]" -s preprocess.resize=64 -s evaluate.bootstrap_samples=20
```

```bash
defectvision package --model baseline_cnn
```

```bash
defectvision reference-stats && defectvision simulate-drift -s monitoring.min_samples_for_drift=20 && defectvision monitor --window 50 -s monitoring.min_samples_for_drift=20
```

`-s key=value` overrides any `params.yaml` key for one invocation without
editing the committed config.

> Accuracy from a smoke run is meaningless — 240 synthetic images at 2 epochs is
> noise. It verifies that the stages connect, nothing more.

---

## 10. Reproducing a specific run

Four things are recorded per run, and all four are needed:

1. **Code** — the git commit (flagged if the tree was dirty)
2. **Config** — content hash + full `params.yaml` snapshot
3. **Data** — the manifest SHA-256
4. **Environment** — interpreter and library versions

Find them in MLflow under the run's `reproducibility/` artifact path, then:

```bash
git checkout <git_commit from the run tags>
```

```bash
dvc repro
```

```bash
defectvision train --model <model_name>
```

`dvc repro` rebuilds only stale stages. Each declares the exact `params.yaml`
keys it reads, so changing `data.synthetic.difficulty` invalidates `acquire` and
everything downstream, while changing `serving.port` invalidates nothing.

Inspect the DAG:

```bash
dvc dag
```

Re-run the entire pipeline through DVC:

```bash
dvc repro --force
```

---

## 11. Troubleshooting

| symptom | cause and fix |
| --- | --- |
| `Python was not found` (Windows) | The Microsoft Store alias stub. Use `py -m venv .venv`, then the venv's own `python`. |
| `Params file not found` | You are not in the repo root. `cd` there, or set `DEFECTVISION_PARAMS`. |
| `Model bundle not found` | No model promoted yet. Run §4 then §5, or `defectvision package --model baseline_cnn`. |
| `/readyz` returns 503 | Working as designed — no model loaded. Same fix as above. The `detail` field says exactly why. |
| `Kaggle API credentials were not found` | Either place `kaggle.json` (see §3a) or set `data.source: synthetic`. |
| `Subset 'casting_data' not found` | `data.kaggle_local_dir` points somewhere without the expected layout. Check §3a, or delete the key to force a download. |
| `Data validation failed on blocking check(s)` | Intentional — the corpus breached a gate. Read `reports/validation_report.json`; the failing check names the reason. |
| `The prediction log is empty` | `monitor` before `simulate-drift`. Run §8b first. |
| `Drift reference not found` | Run `defectvision reference-stats` (§8a). |
| MLflow "filesystem backend is in maintenance mode" | `mlflow.tracking_uri` must be `sqlite:///mlflow.db`, not `file:./mlruns`. MLflow 3.x dropped the file store. |
| Training feels very slow | Expected on CPU: ~40 s/epoch for the baseline, ~150 s for ResNet-18. Use §9 to test the wiring quickly. |
| Port 8000 already in use | `defectvision serve --port 8001` |
| `pytest` fails on a fresh clone | Confirm `pip install -e . --no-deps` ran; `python -c "import defectvision"` should succeed. |

---

## Appendix — one-shot pipeline

**PowerShell** (all of M2 → M5, with progress and timing):

```bash
.\scripts\run_pipeline.ps1
```

**make**:

```bash
make all
```

**Manual sequence**, if you prefer to see each step:

```bash
defectvision data && defectvision train --all && defectvision compare && defectvision reference-stats && defectvision simulate-drift && defectvision monitor && defectvision check-retrain
```

Then, in a second terminal:

```bash
defectvision serve
```
