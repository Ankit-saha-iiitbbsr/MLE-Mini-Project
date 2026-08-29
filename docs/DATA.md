# Data Engineering Notes

**Module M2 · PCAM\* ZC412 Mini-Project (Flavor B)**

Covers where the data comes from, what the validation stage checks and why, and
how the dataset is versioned.

---

## 1. Sources

`data.source` in `params.yaml` selects one of three. All three write the same
layout — `data/raw/<class>/` plus a `_source.json` provenance record — so
nothing downstream knows or cares which was used.

### `kaggle` (default)

The reference corpus: **Casting Product Image Data for Quality Inspection**,
7,348 grayscale 300×300 JPEGs of submersible-pump impellers.

```
datasets/casting_data/casting_data/
├── train/{ok_front, def_front}
└── test/{ok_front, def_front}
```

Two behaviours worth noting:

**A local copy is preferred over a download.** `data.kaggle_local_dir` is
checked first; only if it is absent does the stage authenticate and fetch via
`kagglehub`. Re-downloading 8.6k JPEGs on every clean checkout is slow and
requires credentials that CI does not have.

**The published train/test split is discarded and re-split.** Not because it is
bad, but because the split *policy* — stratification, near-duplicate grouping,
the seed — belongs in `params.yaml` where it is versioned with everything else.
A split you cannot describe in config is a split you cannot reproduce.

### `synthetic`

A procedural generator ([`data/synth.py`](../src/defectvision/data/synth.py))
renders casting-like frames: a light impeller with curved vanes on a dark
background, photographed under varying light and pose, where the defect class
carries small localised flaws (blowholes, pinhole clusters, cracks, chipped
rims, burrs, scratches).

This is not a toy left over from before the real data arrived — it is what makes
the project reproducible on a machine with no Kaggle access, and it is what CI
runs the full DAG against. Two properties were designed in and are enforced by
tests:

1. **Nuisance factors are class-independent.** Brightness, contrast, rotation,
   scale, blur and noise are drawn from the same distribution for both classes,
   so no global statistic can separate them. Measured: all Cohen's *d* < 0.10,
   and a random forest over the seven image statistics scores **0.505** —
   chance.
2. **Defect salience is tunable.** `difficulty` interpolates flaw size and
   contrast. At the committed 0.45, flaws span 6–12 px on a 128 px frame.

Every image is a pure function of `(seed, index)`, so the corpus is regenerable
byte-for-byte from config alone.

### `local`

Trust whatever is already in `data/raw/<class>/`. For dropping in a real
production dump by hand.

---

## 2. Validation

```bash
python -m defectvision.cli validate
```

Severity is split deliberately. **Errors** abort the pipeline; **warnings** are
recorded and the run continues. Making every check fatal trains people to
disable validation; making none fatal means nobody reads it.

### Per-file checks → quarantine, not abort

A production line always produces a few bad frames. Individually bad files are
excluded and recorded; only the *ratio* is fatal.

| check | catches |
| --- | --- |
| format allow-list | stray non-image files in the drop directory |
| `verify()` + decode | truncated or corrupt files |
| dimension bounds | thumbnails, or a resolution change nobody announced |
| file size | empty files, decompression bombs |
| `min_mean_intensity` | all-black frames — lens cap on, lamp failed |
| `max_mean_intensity` | blown-out frames — flash misfire |
| `min_std_intensity` | flat frames carrying no signal at all |

### Corpus-level checks → these abort

| check | severity | why |
| --- | --- | --- |
| `corrupt_file_ratio` | **error** | A few bad files is normal; 5% means the capture system is broken and the quarantine is hiding it |
| `min_images_per_class` | **error** | Too few examples to learn or to evaluate |
| `exact_duplicate_ratio` | **error** | Silently reweights whichever part was photographed twice |
| `class_imbalance` | warning | Handled by class-weighted loss; worth knowing |
| `near_duplicate_ratio` | warning | Handled by grouped splitting; worth knowing |
| `resolution_consistency` | warning | Handled by resize; a *change* here is a signal |
| `global_statistic_leakage` | warning | See below |

### The leakage check

The unusual one. It computes **Cohen's *d*** for every global image statistic
between the two classes and warns if any single statistic nearly separates them
(*d* > 2.0).

If brightness alone separated `ok` from `defect`, then a "defect classifier"
could be a threshold on brightness — it would score beautifully offline and
collapse the first time someone changed a lamp. This check catches that class of
dataset artifact **before** an afternoon is spent training.

Both real corpora pass: no statistic comes close.

### What it found in the real dataset

| finding | count | share | handling |
| --- | --- | --- | --- |
| exact duplicates (SHA-256) | 64 | 0.87% | dropped before splitting |
| near-duplicates (dHash) | 412 | 5.61% | grouped into one fold |
| class imbalance | 4211 : 3137 | 1.34:1 | class-weighted loss |
| corrupt / unreadable | 0 | 0% | — |
| resolution variation | 0 | — | — |

Outputs: `data/interim/scan.csv` (per-file table) and
`reports/validation_report.json` (gate results).

---

## 3. Duplicate detection

Two hashes, because they answer different questions.

**SHA-256 of the file bytes** finds *exact* duplicates — the same file present
twice. These are dropped outright.

**dHash (difference hash)** finds *near* duplicates — the same image re-encoded,
lightly cropped, or the same physical part photographed twice. It resizes to
9×8, compares horizontally adjacent pixels, and packs the 64 comparisons into a
hash. Re-encoding an image as JPEG changes every byte but leaves the dHash
identical, which is exactly the property needed.

Near-duplicates are **not** dropped — they are legitimate data. They are
*grouped*, so every copy of one part lands in the same fold.

### Why this matters more than it sounds

Industrial datasets photograph the same part repeatedly. If copies straddle the
train/test boundary, the model can memorise rather than generalise, and the test
score measures recall of the training set. It is the single most common reason
an image classifier scores 0.99 offline and disappoints in production.

The splitter asserts the invariant and raises if it is ever violated:

```
Leakage check passed: 6872 groups, none spanning folds
```

---

## 4. Splitting

```bash
python -m defectvision.cli split
```

Stratified 65/15/20 with whole groups assigned by a largest-first greedy
bin-packing heuristic — each group goes to whichever fold is furthest below its
quota. Largest-first matters: placing a big group last can overshoot a small
fold badly.

Result on the real corpus — proportions preserved to a tenth of a percent:

| split | n | defect | ok | defect share |
| --- | --- | --- | --- | --- |
| train | 4,734 | 2,737 | 1,997 | 57.8% |
| val | 1,093 | 632 | 461 | 57.8% |
| test | 1,457 | 842 | 615 | 57.8% |

### The output is a file, not an in-memory split

`data/processed/manifest.csv` is the dataset artifact everything downstream
consumes. Making it a file rather than a runtime operation means it is
hashable, reviewable in a diff, and DVC-tracked — which is what makes *"which
images did run #7 train on"* answerable months later. Its SHA-256 is logged to
every MLflow run as the dataset version identifier.

`data/processed/dataset_card.json` carries the counts, the split policy, and the
acquisition provenance.

---

## 5. Preprocessing vs. augmentation

The split between these two is the defence against training–serving skew.

**`preprocess`** — deterministic: grayscale, square resize to 128, tensor
conversion, normalisation. This is the **only** block applied at inference, and
it is **serialised into the model bundle** rather than read from `params.yaml`
at serving time. Editing config after a model ships therefore cannot change how
that model sees an image.

A square resize is used rather than resize-shortest-side plus centre crop,
because a defect near the rim would be cropped away by the latter.

**`augment`** — stochastic, train-only, and built as a wrapper *around* the same
preprocess spec, so there is no second code path that could drift. The policy
mirrors the nuisance factors a real inspection cell produces — placement jitter,
lamp drift, focus wander, sensor noise — which means it doubles as pre-emptive
hardening against the M5 drift scenarios. Corruptions the camera cannot produce
(colour shifts, perspective warps) are deliberately absent.

### Random erasing is off

`random_erasing_p: 0.0`, on purpose. A defect here can occupy fewer than 100 px;
erasing a random patch can delete the only evidence while the label still reads
"defect". That is label noise dressed up as regularisation.

---

## 6. Versioning

Three layers, each answering a different question.

**Git** — code and config. `params.yaml` is committed, so a commit fully
describes the *intent* of a run.

**DVC** — data and models, which are too large for Git. `dvc.yaml` declares the
DAG, and each stage lists the exact `params.yaml` keys it reads:

```bash
dvc repro
```

```bash
dvc dag
```

Because dependencies are declared per-key, changing `data.synthetic.difficulty`
invalidates `acquire` and everything downstream, while changing `serving.port`
invalidates nothing. That precision is what makes `dvc repro` safe to run
habitually instead of being a full rebuild every time.

**MLflow** — run lineage. Each run logs the manifest SHA-256, the git commit
(flagged if the tree was dirty), the config hash, and resolved library versions.

Together they answer: *which code* (git), on *which data* (DVC + manifest hash),
producing *which model* (MLflow run + bundle).

### Setting up a DVC remote

The pipeline works without one — DVC still tracks and caches locally. A remote
is only needed to share artifacts between team members:

```bash
dvc remote add -d storage <url> && dvc push
```

---

## 7. Held-back data

`casting_512x512` (1,300 images) is present in the download but **deliberately
not ingested for training**.

It is the same production line captured by a different camera at a different
resolution — a genuine covariate shift. Held back, it becomes the
`real_camera_upgrade` scenario in the M5 drift simulation: a shift that no
corruption operator was tuned against. Detectors that fire on the synthetic
scenarios but not on this one are overfitted to the simulation.

Folded into training it would have been 1,300 extra rows of near-duplicate
parts. Held back it is the only honest test of whether the monitoring works.
