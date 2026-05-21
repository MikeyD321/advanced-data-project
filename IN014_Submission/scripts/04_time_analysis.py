#!/usr/bin/env python3
"""Resolution time analysis: summaries, plots, ANOVA."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

from _paths import CLEANED, FIGURES, METRICS, ensure_output_dirs


def anova(groups: list, name: str) -> dict:
    if len(groups) < 2:
        return {"factor": name, "skipped": True}
    f, p = stats.f_oneway(*groups)
    return {"factor": name, "f": float(f), "p_value": float(p), "significant_05": bool(p < 0.05)}


def main() -> None:
    ensure_output_dirs()
    sns.set_theme(style="whitegrid")
    df = pd.read_csv(CLEANED / "tickets_enriched.csv", low_memory=False)
    df["resolution_time_hours"] = pd.to_numeric(df["resolution_time_hours"], errors="coerce")
    resolved = df[df["resolution_time_hours"].notna()].copy()

    summary = {}
    anova_results = []
    for col in ("priority", "sla_plan", "region"):
        if col not in resolved.columns:
            continue
        grp = resolved.groupby(col)["resolution_time_hours"].agg(["mean", "median", "std", "count"])
        summary[col] = grp.reset_index().to_dict(orient="records")
        print(f"\n--- Mean resolution hours by {col} ---\n{grp.round(2)}")
        groups = [g["resolution_time_hours"].values for _, g in resolved.groupby(col) if len(g) >= 10]
        anova_results.append(anova(groups, col))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col in zip(axes, ("priority", "sla_plan", "region")):
        order = resolved.groupby(col)["resolution_time_hours"].median().sort_values().index
        sns.boxplot(data=resolved, x=col, y="resolution_time_hours", order=order, ax=ax)
        ax.set_title(f"Resolution time by {col}")
        ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    box_path = FIGURES / "resolution_time_boxplots.png"
    fig.savefig(box_path, dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=resolved, x="priority", y="resolution_time_hours", ax=ax, cut=0)
    ax.set_title("Resolution time by priority (violin)")
    v_path = FIGURES / "resolution_time_violin.png"
    fig.tight_layout()
    fig.savefig(v_path, dpi=120)
    plt.close(fig)

    metrics = {
        "n_resolved": len(resolved),
        "mean_hours": float(resolved["resolution_time_hours"].mean()),
        "median_hours": float(resolved["resolution_time_hours"].median()),
        "summary_by_group": summary,
        "anova": anova_results,
        "figures": [box_path.name, v_path.name],
    }
    with open(METRICS / "04_time_analysis.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== ANOVA ===")
    for r in anova_results:
        if r.get("skipped"):
            print(f"  {r['factor']}: skipped")
        else:
            print(f"  {r['factor']}: F={r['f']:.2f}, p={r['p_value']:.2e}, sig@0.05={r['significant_05']}")

    print("\n[COMPLETE] 04_time_analysis.py")


if __name__ == "__main__":
    main()
