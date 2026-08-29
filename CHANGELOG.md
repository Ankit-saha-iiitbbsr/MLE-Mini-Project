# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2026-08-15

First complete end-to-end system. Covers modules M2 through M5 of the
PCAM\* ZC412 mini-project (Flavor B — Image-Based Defect / Quality Classifier).

### M2 · Data engineering

**Added**

- Three interchangeable data sources behind one contract (`data/raw/<class>/`
  plus a provenance record): the real Kaggle casting corpus, a procedural
  generator, and a local drop-in directory.
- Procedural casting generator (`data/synth.py`) — renders impeller frames with
  six defect types and class-independent nuisance factors. Keeps the whole
  pipeline reproducible with no credentials, which is what lets CI exercise the
  full DAG.
- Validation stage with split severity: per-file checks quarantine bad frames,
  corpus-level checks abort the run.
- Duplicate detection on two hashes — SHA-256 for exact copies (dropped) and
  dHash for near-duplicates (grouped, not dropped).
- Shortcut-leakage check: computes Cohen's *d* for every global image statistic
  between classes and warns if any single one nearly separates them.
- Leakage-safe stratified splitter; asserts that no group spans two folds.
- Manifest + dataset card as the versioned dataset artifact.
- Seven-statistic feature module shared by training and serving.
- DVC pipeline with per-key parameter dependencies.

**Found in the real corpus** — 64 exact duplicates, 412 near-duplicates,
1.34:1 class imbalance; 0 corrupt files.

### M3 · Experimentation

**Added**

- Four competing arms: baseline CNN, ResNet-18, MobileNetV3-Small, and a
  HOG + logistic-regression classical control.
- Grayscale stem adaptation for transfer models — folds pretrained RGB filter
  weights into a 1-channel conv, response-identical to channel repetition at a
  third of the cost (asserted by test).
- MLflow tracking of config, metrics, plots, latency profile and bundle.
- Reproducibility capture: git commit (dirty-flagged), config hash, dataset
  manifest SHA-256, resolved library versions.
- Threshold tuning on validation with three strategies; test touched once.
- Bootstrap confidence intervals on all headline test metrics.
- Promotion gates plus ranking, with a latency tie-break inside the leader's
  confidence interval.
- MLflow Model Registry integration.

**Notable**

The baseline CNN sat at exactly chance (0.50 accuracy) until the pooling head
was changed. A defect covers under 1% of the frame, so global average pooling
divides that evidence across the whole feature map; concatenating global **max**
pooling asks the question the task actually poses. This is recorded because it
was the single decision that made the model work at all.

### M4 · Packaging and serving

**Added**

- Self-contained model bundle carrying weights, preprocessing spec, class
  ordering and the tuned threshold. The serving container needs no MLflow.
- FastAPI service: `/predict`, `/predict/batch`, `/healthz`, `/readyz`,
  `/model`, `/metrics`, `/feedback`.
- Separate liveness and readiness endpoints.
- Typed 4xx responses with actionable hints for every malformed-input path;
  a 5xx is reserved for genuine bugs.
- Partially fault-tolerant batch: one corrupt file does not discard the rest.
- Human-review band for borderline confidences.
- End-to-end HTTP load benchmark (not `model.forward()` timing).
- Multi-stage Dockerfile on CPU-only torch, non-root, with a readiness
  healthcheck; plus docker-compose with the MLflow UI.

### M5 · Monitoring, drift and retraining

**Added**

- SQLite prediction log carrying decision, image statistics, model identity and
  nullable delayed ground truth.
- PSI (frozen reference bin edges), two-sample KS, and chi-square detectors.
- Drift simulation over physically-motivated corruptions, plus a `baseline`
  control establishing the noise floor.
- `real_camera_upgrade` scenario built from the `casting_512x512` capture, held
  back from training as a genuine covariate shift.
- Monitoring report with PSI heatmap, per-scenario performance, rolling-window
  trend, and score distributions.
- Retraining trigger with persistence, cooldown and label-sufficiency gating,
  and severity routing that separates `retrain` from `recalibrate` from
  `investigate_capture`.

### Project infrastructure

**Added**

- Unified `defectvision` CLI covering the whole lifecycle; DVC stages shell out
  to the same commands a developer runs by hand.
- 127 tests against a temp-directory synthetic corpus.
- GitHub Actions CI: lint, tests on Python 3.10/3.12, a full synthetic pipeline
  run asserting cross-stage contracts, and a Docker build that verifies the
  container reports *not ready* without a model.
- Makefile task shortcuts.
- Documentation: README with architecture diagram, `docs/DATA.md`,
  `docs/RETRAINING_DESIGN.md`, `docs/API_EXAMPLES.md`, `docs/DEMO_SCRIPT.md`,
  and a Postman collection.

### Security

- The Kaggle integration authenticates with an **API token** only, read from the
  standard `kaggle.json` location or environment variables. Account passwords
  are never accepted, requested, or logged. `kaggle.json` is git-ignored.
- The serving container runs as a non-root user and never writes to the mounted
  model directory.
- Upload decoding is defensive: size limits, format allow-list, and a Pillow
  decompression-bomb guard.
