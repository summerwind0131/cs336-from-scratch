import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.markers import MarkerStyle


# 1. 把算力预算相同的实验放在一起。
runs = json.loads(Path("data/isoflops_curves.json").read_text())
groups = defaultdict(list)
for run in runs:
    groups[run["compute_budget"]].append(run)


# 2. 每档预算选出 loss 最低的实验，得到 9 个观测最优点。
budgets = np.array(sorted(groups))
best = [min(groups[c], key=lambda r: r["final_loss"]) for c in budgets]
params = np.array([r["parameters"] for r in best])
tokens = budgets / (6 * params)

# 3. 画出最小预算下的完整曲线。红星是这一档选中的实验。
fig, ax = plt.subplots()
profile = sorted(groups[budgets[0]], key=lambda r: r["parameters"])
ax.semilogx(
    [r["parameters"] for r in profile],
    [r["final_loss"] for r in profile],
    "o-",
)
ax.plot(params[0], best[0]["final_loss"], "r*", markersize=14)
ax.set(
    xlabel="Parameters",
    ylabel="Final training loss",
    title=f"IsoFLOPs profile at {budgets[0]:.0e} FLOPs",
)
fig.savefig("isoflops_example.png", dpi=160, bbox_inches="tight")
plt.close(fig)


# 4. 分别拟合 N_opt(C) 和 D_opt(C)，并外推到题目给的两个预算。
#    如果 y = k * C**a，那么 log10(y) = log10(k) + a * log10(C)。
targets = np.array([1e23, 1e24])
grid = np.geomspace(budgets[0], targets[-1], 200)

for values, label, filename in [
    (params, "Parameters", "optimal_params.png"),
    (tokens, "Tokens", "optimal_tokens.png"),
]:
    slope, intercept = np.polyfit(np.log10(budgets), np.log10(values), 1)
    predictions = 10 ** (intercept + slope * np.log10(targets))

    print(f"{label} exponent: {slope:.4f}")
    for c, value in zip(targets, predictions):
        print(f"  {c:.0e} FLOPs -> {value:.3e} {label.lower()}")

    fig, ax = plt.subplots()
    ax.loglog(
        grid,
        10 ** (intercept + slope * np.log10(grid)),
        label=f"Power law (exponent {slope:.3f})",
    )
    ax.scatter(budgets, values, label="Observed optima", zorder=3)
    ax.scatter(targets, predictions, marker=MarkerStyle("x"), label="Predictions", zorder=3)
    ax.set(xlabel="Compute budget (FLOPs)", ylabel=label)
    ax.legend()
    fig.savefig(filename, dpi=160, bbox_inches="tight")
    plt.close(fig)
