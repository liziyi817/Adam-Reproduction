import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="Plot logged optimizer runs.")
parser.add_argument("--results", type=Path, default=Path("results"))
args = parser.parse_args()
root = args.results
data = json.loads((root / "metrics.json").read_text())
runs = data["runs"]
styles = [("sgd", "Momentum SGD", "#9a432a"),
          ("adam", "Adam", "#245b93"),
          ("uncorrected", "Adam without correction", "#31806c")]
fig, axes = plt.subplots(1, 2, figsize=(8, 3.25), layout="constrained")
for ax, key, label in zip(axes, ("loss", "accuracy"), ("Train cross-entropy", "Train accuracy (%)")):
    for mode, name, color in styles:
        a = np.array([[e[key] for e in r["history"]] for r in runs if r["model"] == mode])
        if key == "accuracy":
            a *= 100
        epochs = np.arange(1, a.shape[1]+1)
        ax.plot(epochs, a.mean(0), label=name, color=color)
        ax.fill_between(epochs, a.min(0), a.max(0), alpha=.12, color=color)
    ax.grid(alpha=.2)
    ax.set(xlabel="Epoch", ylabel=label, xticks=[1,5,10,15,20])
axes[0].legend(frameon=False, fontsize=8)
fig.savefig(root / "learning.png", dpi=230)
plt.close(fig)
fig, ax = plt.subplots(figsize=(5.3, 2.8), layout="constrained")
for mode, name, color in styles[:2]:
    vals = data["tuning"][mode]
    rate = [v["lr"] for v in vals]
    values = 100 * np.array([v["accuracy"] for v in vals])
    ax.plot(rate, values.mean(1), "o-", label=name, color=color)
    ax.fill_between(rate, values.min(1), values.max(1), color=color, alpha=.12)
ax.set(xscale="log", xlabel="Learning rate", ylabel="Validation accuracy (%)")
ax.grid(alpha=.2)
ax.legend(frameon=False)
fig.savefig(root / "tuning.png", dpi=230)
