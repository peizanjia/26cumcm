"""Create a transparent provisional comparison with commit 56d71c4.

The new checkpoint and raw evaluation files are not present in the repository,
so its values below are transcribed from the commit-pinned radius20 README.
Local minimax values are always read from results/summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).parent
COMMIT = "56d71c42248d5a9019bfb0100d4233a2f5a3f5a2"
SOURCE = (
    "https://github.com/peizanjia/26cumcm/blob/"
    + COMMIT
    + "/question2/%E9%BB%91%E7%AE%B1%E6%80%9D%E8%B7%AF/radius20/README.md"
)

PUBLISHED = {
    "new_nn_plain_abs_radius20": {
        "events": 1_048_576,
        "mean_mec_radius_m": 24.23972,
        "p95_mec_radius_m": 37.79959,
        "max_mec_radius_m": 750.11425,
        "mec_mae20_m": 5.15709,
        "mec_success_le20": 0.31098,
        "no_signal_rate": 0.0002794,
    },
    "new_nn_orthogonal_lambda_0.1": {
        "events": 1_048_576,
        "mean_mec_radius_m": 24.22452,
        "p95_mec_radius_m": 37.89499,
        "max_mec_radius_m": 750.11425,
        "mec_mae20_m": 5.16951,
        "mec_success_le20": 0.31161,
        "no_signal_rate": 0.0003090,
    },
}


def main() -> None:
    local = json.loads((ROOT / "results" / "summary.json").read_text())
    minimax = next(
        row for row in local["summary"] if row["policy"] == "mechanistic_minimax"
    )
    old_nn = next(
        row for row in local["summary"] if row["policy"] == "repository_neural"
    )
    selected = {
        "mechanistic_minimax": {
            "events": minimax["trials"],
            **{key: minimax[key] for key in (
                "mean_mec_radius_m", "p95_mec_radius_m", "max_mec_radius_m",
                "mec_mae20_m", "mec_success_le20", "no_signal_rate",
                "mean_diameter_m", "p95_diameter_m", "max_diameter_m",
                "certified_reception_rate",
            )},
        },
        "old_area_nn": {
            "events": old_nn["trials"],
            **{key: old_nn[key] for key in (
                "mean_mec_radius_m", "p95_mec_radius_m", "max_mec_radius_m",
                "mec_mae20_m", "mec_success_le20", "no_signal_rate",
                "mean_diameter_m", "p95_diameter_m", "max_diameter_m",
                "certified_reception_rate",
            )},
        },
        **PUBLISHED,
    }
    for name in PUBLISHED:
        radius = selected[name]
        radius["diameter_bounds_from_mec_m"] = {
            "mean": [radius["mean_mec_radius_m"], 2 * radius["mean_mec_radius_m"]],
            "p95": [radius["p95_mec_radius_m"], 2 * radius["p95_mec_radius_m"]],
            "max": [radius["max_mec_radius_m"], 2 * radius["max_mec_radius_m"]],
        }

    payload = {
        "status": "provisional_unpaired_comparison",
        "new_commit": COMMIT,
        "published_source": SOURCE,
        "metric_correction": (
            "The new primary loss is E[abs(MEC_radius-20)]/20, not polygon diameter."
        ),
        "missing_from_commit": [
            "question2/黑箱思路/radius20/checkpoints/plain.pt",
            "question2/黑箱思路/radius20/checkpoints/orthogonal.pt",
            "question2/黑箱思路/radius20/results/evaluation.json",
            "question2/黑箱思路/radius20/results/evaluation_samples.npz",
        ],
        "comparability": (
            "Minimax uses 128 locally simulated one-event states and official second-reception "
            "geometry; new-NN numbers are author-reported over 2048 states x 256 events x "
            "two equally weighted sides and omit the second 1500 m feasible-set disk."
        ),
        "policies": selected,
    }
    (ROOT / "results" / "new_nn_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    names = ["Minimax\n(local)", "Old area NN\n(local)",
             "New radius20\n(published)", "New guided\n(published)"]
    rows = [selected[k] for k in (
        "mechanistic_minimax", "old_area_nn", "new_nn_plain_abs_radius20",
        "new_nn_orthogonal_lambda_0.1")]
    metrics = [
        ("mec_mae20_m", "Mean |MEC radius - 20| (m)", False),
        ("mean_mec_radius_m", "Mean MEC radius (m)", False),
        ("p95_mec_radius_m", "P95 MEC radius (m)", False),
        ("max_mec_radius_m", "Maximum MEC radius (m, log scale)", True),
    ]
    colors = ["#2a9d8f", "#718096", "#e76f51", "#f4a261"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    x = np.arange(len(names))
    for axis, (key, title, log_scale) in zip(axes.ravel(), metrics):
        values = [row[key] for row in rows]
        bars = axis.bar(x, values, color=colors)
        axis.set_xticks(x, names, fontsize=8)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22)
        if log_scale:
            axis.set_yscale("log")
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value,
                      f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Minimax vs new radius20 NN — provisional, non-paired comparison")
    fig.tight_layout(rect=(0.0, 0.075, 1.0, 0.96))
    fig.text(0.5, 0.022,
             "Local: 128 trials. Published NN: 1,048,576 event-sides. "
             "Weights/raw outputs absent from commit; do not interpret bars as a paired test.",
             ha="center", fontsize=8, color="#555555")
    fig.savefig(ROOT / "results" / "new_nn_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
