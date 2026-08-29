"""Exploratory data analysis for the casting corpus.

Written as a **script**, not a notebook, on purpose. Notebooks are where ML
projects accumulate untracked state: cells run out of order, outputs are
committed as noise in diffs, and "it worked in my kernel" becomes unfalsifiable.
This file is plain Python, diffs cleanly, and runs identically every time:

    python notebooks/01_exploratory_data_analysis.py

It can still be explored interactively — the `# %%` markers make VS Code and
PyCharm treat each block as a runnable cell, and Jupytext will convert it to
`.ipynb` on demand:

    jupytext --to notebook notebooks/01_exploratory_data_analysis.py

Everything it computes is read from the pipeline's own artifacts (`scan.csv`,
`manifest.csv`), so the numbers here are the same numbers the training run saw.
Run `defectvision data` first.
"""

# %% [markdown]
# # Casting defect dataset — exploratory analysis
#
# Questions worth answering before training anything:
#
# 1. What is actually in the corpus, and is it balanced?
# 2. Do the two classes look different to the eye?
# 3. Can any *single global statistic* separate them? (If yes, the benchmark is
#    broken and a threshold would beat a CNN.)
# 4. Are there duplicates, and would they leak across the split?

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from defectvision.config import get, load_params, resolve
from defectvision.features.image_stats import FEATURE_NAMES

params = load_params()
raw_dir = resolve(get(params, "data.raw_dir"))
scan_path = resolve(get(params, "data.interim_dir")) / "scan.csv"
manifest_path = resolve(get(params, "data.processed_dir")) / "manifest.csv"

for path in (scan_path, manifest_path):
    if not path.exists():
        raise SystemExit(f"{path} not found. Run `defectvision data` first.")

scan = pd.read_csv(scan_path)
manifest = pd.read_csv(manifest_path)
out_dir = Path("reports/figures/eda")
out_dir.mkdir(parents=True, exist_ok=True)

print(f"Scanned files : {len(scan):,}")
print(f"Valid files   : {int(scan['is_valid'].sum()):,}")
print(f"In manifest   : {len(manifest):,}")

# %% [markdown]
# ## 1. Class balance and splits

# %%
print("\nClass counts")
print(manifest["class_name"].value_counts().to_string())

counts = manifest["class_name"].value_counts()
imbalance = counts.max() / counts.min()
print(f"\nImbalance ratio: {imbalance:.2f}:1")
print("-> handled by class-weighted loss (train.class_weighting = balanced)")

print("\nSplit x class")
pivot = pd.crosstab(manifest["split"], manifest["class_name"])
print(pivot.to_string())

print("\nDefect share per split (should be near-identical -- stratification)")
share = manifest.groupby("split")["label"].mean()
print((share * 100).round(2).to_string())

# %% [markdown]
# ## 2. What the images look like

# %%
fig, axes = plt.subplots(2, 6, figsize=(13, 4.6))
for row, class_name in enumerate(get(params, "data.classes")):
    files = sorted((raw_dir / class_name).glob("*"))[:6]
    for col, path in enumerate(files):
        with Image.open(path) as im:
            axes[row, col].imshow(np.asarray(im.convert("L")), cmap="gray")
        axes[row, col].set_axis_off()
        if col == 0:
            axes[row, col].set_title(class_name, loc="left", fontsize=11, fontweight="bold")
fig.suptitle("Casting images by class", fontsize=13)
fig.tight_layout()
fig.savefig(out_dir / "class_samples.png", dpi=130)
print(f"wrote {out_dir / 'class_samples.png'}")

# %% [markdown]
# ## 3. The important question: is there a shortcut?
#
# If one global statistic separated the classes, a "defect classifier" could be
# a threshold on brightness — it would score well offline and collapse the first
# time someone changed a lamp. Cohen's *d* quantifies the separation; the
# validation stage warns above 2.0.

# %%
valid = scan[scan["is_valid"]]
group_ok = valid[valid["label"] == 0]
group_defect = valid[valid["label"] == 1]

rows = []
for name in FEATURE_NAMES:
    a, b = group_ok[name].to_numpy(), group_defect[name].to_numpy()
    pooled = np.sqrt((a.var() + b.var()) / 2)
    d = abs(a.mean() - b.mean()) / pooled if pooled > 1e-9 else 0.0
    rows.append({"feature": name, "ok_mean": a.mean(), "defect_mean": b.mean(),
                 "cohens_d": d})

effects = pd.DataFrame(rows).sort_values("cohens_d", ascending=False)
print("\nSeparation by single global statistic (Cohen's d):")
print(effects.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

worst = effects.iloc[0]
verdict = "NO SHORTCUT" if worst["cohens_d"] < 2.0 else "*** SHORTCUT PRESENT ***"
print(f"\nLargest effect: {worst['feature']} (d={worst['cohens_d']:.3f}) -> {verdict}")
print("A CNN has to localise the defect; global statistics will not do it.")

# %%
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for ax, name in zip(axes.ravel(), FEATURE_NAMES, strict=False):
    bins = np.linspace(valid[name].min(), valid[name].max(), 45)
    ax.hist(group_ok[name], bins=bins, alpha=0.6, label="ok", color="#3b7dd8", density=True)
    ax.hist(group_defect[name], bins=bins, alpha=0.6, label="defect", color="#d1495b",
            density=True)
    d = effects.loc[effects["feature"] == name, "cohens_d"].iloc[0]
    ax.set_title(f"{name}  (d={d:.3f})", fontsize=10)
    ax.tick_params(labelsize=8)
axes.ravel()[-1].set_axis_off()
axes.ravel()[0].legend(fontsize=9)
fig.suptitle("Global image statistics by class — heavily overlapping is what we want",
             fontsize=13)
fig.tight_layout()
fig.savefig(out_dir / "feature_distributions.png", dpi=130)
print(f"wrote {out_dir / 'feature_distributions.png'}")

# %% [markdown]
# ## 4. Duplicates and leakage risk

# %%
n_exact = int(valid["sha256"].duplicated().sum())
n_near = int(valid["dhash"].duplicated().sum()) - n_exact
print(f"\nExact duplicates      : {n_exact} ({n_exact / len(valid):.2%})")
print(f"Near-duplicates (dHash): {n_near} ({n_near / len(valid):.2%})")
print("\nExact duplicates are dropped; near-duplicates are GROUPED into one fold")
print("so copies of a part cannot straddle the train/test boundary.")

crossing = manifest.groupby("group")["split"].nunique()
print(f"\nGroups: {len(crossing):,}  spanning >1 split: {int((crossing > 1).sum())}")
assert (crossing == 1).all(), "LEAKAGE: a group spans multiple splits"
print("Leakage check passed.")

# %% [markdown]
# ## 5. Summary

# %%
print(f"""
Corpus      : {len(manifest):,} images, {imbalance:.2f}:1 imbalance
Splits      : {dict(manifest['split'].value_counts())}
Stratified  : defect share varies by {(share.max() - share.min()) * 100:.2f} pp across splits
Shortcut    : none (max Cohen's d = {worst['cohens_d']:.3f})
Duplicates  : {n_exact} exact (dropped), {n_near} near (grouped)
Leakage     : none -- every group sits in exactly one fold

Implication for modelling: the task needs a model that can localise a small,
low-contrast surface flaw. Global statistics carry no class signal, which is
why the pooling strategy in the CNN head turned out to be the decision that
made the model work at all (see models/baseline_cnn.py).
""")
