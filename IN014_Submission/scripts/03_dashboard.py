#!/usr/bin/env python3
"""Build operational dashboard (Plotly HTML + static PNGs)."""
from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from _paths import CLEANED, FIGURES, METRICS, ensure_output_dirs


def merge_dates(fact: pd.DataFrame, enr: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("ticket_id", "created_at", "customer_segment") if c in enr.columns]
    out = fact.merge(enr[cols].drop_duplicates("ticket_id"), on="ticket_id", how="left")
    out["created_at"] = pd.to_datetime(out["created_at"], errors="coerce")
    return out


def sla_met(df: pd.DataFrame) -> pd.DataFrame:
    limits = {"standard": 48, "premium": 24, "gold": 24, "platinum": 8}
    r = df[df["resolution_time_hours"].notna()].copy()
    r["limit"] = r["sla_plan"].map(limits).fillna(48)
    r["met"] = r["resolution_time_hours"] <= r["limit"]
    return r.groupby("customer_segment", as_index=False)["met"].mean().rename(columns={"met": "sla_rate"})


def plotly_dashboard(fact: pd.DataFrame, enr: pd.DataFrame) -> str | None:
    if not HAS_PLOTLY:
        return None
    fact = merge_dates(fact, enr)
    fact["month"] = fact["created_at"].dt.to_period("M").astype(str)

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Tickets by priority (monthly)", "Resolution time (hours)",
            "SLA compliance by segment", "Channel mix",
            "Sentiment", "CSAT (surveys only)",
        ),
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "domain"}], [{"type": "domain"}, {"type": "xy"}]],
        vertical_spacing=0.1,
    )
    vol = fact.groupby(["month", "priority"]).size().reset_index(name="n")
    for p in vol["priority"].unique():
        s = vol[vol["priority"] == p]
        fig.add_trace(go.Scatter(x=s["month"], y=s["n"], mode="lines+markers", name=str(p)), row=1, col=1)

    res = fact[fact["resolution_time_hours"].notna()]
    fig.add_trace(go.Histogram(x=res["resolution_time_hours"], nbinsx=40), row=1, col=2)

    sla = sla_met(enr)
    fig.add_trace(go.Bar(x=sla["customer_segment"], y=sla["sla_rate"]), row=2, col=1)

    ch = fact["channel"].value_counts()
    fig.add_trace(go.Pie(labels=ch.index, values=ch.values), row=2, col=2)

    sent = fact["customer_sentiment"].value_counts()
    fig.add_trace(go.Pie(labels=sent.index, values=sent.values), row=3, col=1)

    csat = fact[fact["csat_score"] > 0]
    fig.add_trace(go.Histogram(x=csat["csat_score"], nbinsx=5), row=3, col=2)

    fig.update_layout(height=1000, width=1100, title_text="IT Support Dashboard", template="plotly_white")
    path = FIGURES / "dashboard.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    return str(path)


def static_plots(fact: pd.DataFrame, enr: pd.DataFrame) -> list[str]:
    sns.set_theme(style="whitegrid")
    saved = []
    fact = merge_dates(fact, enr)
    fact["month"] = fact["created_at"].dt.to_period("M").astype(str)

    fig, ax = plt.subplots(figsize=(10, 5))
    fact.groupby(["month", "priority"]).size().unstack(fill_value=0).plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Ticket volume by priority over time")
    ax.tick_params(axis="x", rotation=45)
    p = FIGURES / "volume_by_priority.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    saved.append(p.name)

    res = fact[fact["resolution_time_hours"].notna()]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(res["resolution_time_hours"], bins=40, kde=True, ax=ax)
    ax.set_title("Resolution time distribution")
    p = FIGURES / "resolution_time_dist.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    saved.append(p.name)

    sla = sla_met(enr)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=sla, x="customer_segment", y="sla_rate", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("SLA compliance by segment")
    p = FIGURES / "sla_by_segment.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    saved.append(p.name)

    return saved


def main() -> None:
    ensure_output_dirs()
    fact = pd.read_csv(CLEANED / "fact_tickets.csv", low_memory=False)
    enr = pd.read_csv(CLEANED / "tickets_enriched.csv", low_memory=False, usecols=lambda c: c in {
        "ticket_id", "created_at", "customer_segment", "resolution_time_hours", "sla_plan", "channel",
        "customer_sentiment", "csat_score", "priority", "region",
    } or True)
    if "created_at" not in enr.columns:
        enr = pd.read_csv(CLEANED / "tickets_enriched.csv", low_memory=False)

    html = plotly_dashboard(fact, enr)
    if html:
        print(f"Dashboard HTML: {html}")
    pngs = static_plots(fact, enr)
    print(f"Saved PNGs: {', '.join(pngs)}")

    with open(METRICS / "03_dashboard.json", "w") as f:
        json.dump({"html": "outputs/charts/dashboard.html", "figures": pngs, "n_tickets": len(fact)}, f, indent=2)
    print("\n[COMPLETE] 03_dashboard.py")


if __name__ == "__main__":
    main()
