#!/usr/bin/env python3
"""Generate submission report.md from pipeline metrics and figures."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from _paths import FIGURES, METRICS, REPORTS, ROOT, ensure_output_dirs

REPORT = ROOT / "report.md"


def load_json(name: str) -> dict:
    p = METRICS / name
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def fig(name: str, caption: str) -> str:
    rel = f"outputs/charts/{name}"
    if (FIGURES / name).exists() or (ROOT / rel).exists():
        return f"![{caption}]({rel})\n\n*{caption}*\n\n"
    return f"<!-- missing: {rel} -->\n\n"


def main() -> None:
    ensure_output_dirs()
    fix = load_json("02_fix_quality.json")
    dash = load_json("03_dashboard.json")
    time_a = load_json("04_time_analysis.json")
    csat = load_json("05_csat_model.json")
    res = load_json("06_resolution_prediction.json")
    sent = load_json("07_sentiment_classifier.json")
    agent = load_json("08_agent_reply_demo.json")
    clean = load_json("01_cleaning_stats.json")  # may not exist if only 01 metrics names differ

    ts = datetime.now().strftime("%Y-%m-%d")

    body = f"""---
title: "IN014 Final Project — IT Support Ticket Analytics"
---

# IN014 — Advanced Data Processing and Analysis

**IT Support Ticket Analytics Pipeline**

| | |
|---|---|
| **Authors** | <NAMES> |
| **Course** | <COURSE_NAME> |
| **Date** | <DATE> |
| **Generated** | {ts} |

## Table of Contents

1. [Introduction](#1-introduction)
2. [Data governance](#2-data-governance)
3. [Data stack](#3-data-stack)
4. [Data model](#4-data-model)
5. [Findings and results](#5-findings-and-results)
6. [Reflection](#6-reflection)
7. [Appendix](#appendix)

## Table of figures

- dashboard.html / volume_by_priority.png
- resolution_time_boxplots.png
- csat_feature_importance.png
- resolution_pred_scatter.png
- sentiment_confusion_matrix.png

---

## 1. Introduction

We analyze **100,000 synthetic IT support tickets** paired with a customer master file. Each row is a ticket with timestamps, channel, product area, priority, SLA tier, free-text messages, resolution outcomes, sentiment labels, and CSAT scores.

**Business problem:** Support leaders need to see where volume and delays concentrate, which customers are unhappy, and whether we can predict satisfaction or resolution time from ticket text and metadata. This project builds a reproducible pipeline: clean data → star schema → dashboard → statistical analysis → supervised models.

**Data source:** Course-provided CSVs (`it_support_tickets.csv`, `customer_info.csv`) stored in `data/raw/`.

---

## 2. Data governance

| Topic | Approach |
|-------|----------|
| **Ownership** | Synthetic course dataset; treat messages as potentially PII |
| **Quality rules** | `resolution_time_hours` NaN = open ticket; `csat_score` 0 = no survey |
| **Duplicates** | Exact duplicate ticket rows dropped in cleaning; `customer_id` deduped in dim |
| **Lineage** | `data/raw` → `01_clean_data.py` → `data/cleaned` → analysis scripts → `outputs/` |

**Known limitation:** Only ~1,046 ticket `customer_id` values appear in `customer_info.csv` (~9,953 ticket customers have no master record). Country-based region imputation only helps tickets with a matching customer row.

---

## 3. Data stack

We use a **local Python stack** (pandas, scikit-learn, matplotlib, seaborn, plotly) rather than a cloud warehouse. At 100k rows everything fits in memory on a laptop, which keeps the project reproducible for grading.

| Layer | Tool |
|-------|------|
| ETL | pandas, pathlib |
| Storage | CSV in `data/cleaned/` |
| Analytics | scikit-learn, scipy |
| Viz | matplotlib, seaborn, plotly (HTML dashboard) |
| Orchestration | `run_all.sh` |

A production deployment could move the same logic to BigQuery/Snowflake + Airflow; the star schema maps cleanly to a warehouse model.

---

## 4. Data model

**Cleaning (`01_clean_data.py`):** validate columns, parse dates, binary flags, merge customers, build star schema.

**Quality fixes (`02_fix_quality.py`):** dedupe `dim_customer`, add `csat_response`, impute `region` from `country`.

| Table | Role |
|-------|------|
| `fact_tickets` | One row per ticket; measures + dimensions |
| `dim_customer` | Customer attributes |
| `dim_date` | Calendar |
| `dim_channel` | Channels |
| `dim_product_area` | Product areas |
| `tickets_enriched` | Wide table for ML |

```
data/raw → 01_clean → 02_fix → data/cleaned → 03–08 analysis → outputs/
```

---

## 5. Findings and results

### 5.1 Data quality

- Duplicate customers removed: **{fix.get('duplicate_customers_removed', 'N/A')}**
- Regions imputed from country: **{fix.get('regions_imputed_fact', 'N/A'):,}** (fact table)
- Regions set to Unknown when country unmappable: **{fix.get('regions_unknown_fact', 'N/A'):,}**

### 5.2 Dashboard

{fig('volume_by_priority.png', 'Ticket volume by priority over time')}
{fig('resolution_time_dist.png', 'Resolution time distribution')}
{fig('sla_by_segment.png', 'SLA compliance by customer segment')}

Open `outputs/charts/dashboard.html` for the interactive view.

### 5.3 Resolution time analysis

- Resolved tickets analyzed: **{time_a.get('n_resolved', 'N/A'):,}**
- Mean resolution time: **{time_a.get('mean_hours', 0):.1f} hours** (median **{time_a.get('median_hours', 0):.1f}**)

{fig('resolution_time_boxplots.png', 'Resolution time by priority, SLA, and region')}

ANOVA suggests resolution times differ significantly across priority, SLA plan, and region (see `outputs/metrics/04_time_analysis.json`).

### 5.4 CSAT prediction

Random forest on TF-IDF + categoricals (survey respondents only):

- **RMSE:** {csat.get('rmse', 'N/A')}
- **R²:** {csat.get('r2', 'N/A')}

{fig('csat_feature_importance.png', 'CSAT model feature importance')}

### 5.5 Resolution time prediction

- **RMSE:** {res.get('rmse_hours', 'N/A')} hours
- **MAE:** {res.get('mae_hours', 'N/A')} hours
- **R²:** {res.get('r2', 'N/A')}

{fig('resolution_pred_scatter.png', 'Actual vs predicted resolution hours')}

### 5.6 Sentiment classification

- **Accuracy:** {sent.get('accuracy', 0):.3f}
- **Macro F1:** {sent.get('macro_f1', 0):.3f}

{fig('sentiment_confusion_matrix.png', 'Sentiment confusion matrix')}

### 5.7 Agent reply generation (demo)

Template / optional FLAN-T5 demo — see `outputs/samples/agent_reply_examples.txt`. Mode: **{agent.get('mode', 'N/A')}**.

---

## 6. Reflection

We learned that **most project time goes into data understanding**, not modeling. The customer file overlap issue materially affects enrichment and region imputation — reporting that honestly matters more than chasing a high R² on a leaky feature.

Simple models (logistic regression, random forest + TF-IDF) were enough to show feasibility. Sentiment labels are synthetic; real deployments would need human-labeled training data.

**For teammates:** See README “Future work” — interactive Streamlit dashboard, XGBoost tuning, LLM fine-tuning for replies, and fairness checks by segment.

---

## Appendix

### Reproduce

```bash
pip install -r requirements.txt
bash run_all.sh
```

### Scripts

| Script | Purpose |
|--------|---------|
| `01_clean_data.py` | ETL + star schema |
| `02_fix_quality.py` | DQ fixes |
| `03_dashboard.py` | Charts |
| `04_time_analysis.py` | ANOVA + boxplots |
| `05_csat_model.py` | CSAT regression |
| `06_resolution_prediction.py` | Resolution regression |
| `07_sentiment_classifier.py` | Sentiment ML |
| `08_agent_reply_demo.py` | Reply demo |
| `generate_report.py` | This report |

### GitHub

<NAMES>: add repository URL here. Do not commit large raw CSVs — link to course download or use Git LFS.

---

*Fill in <NAMES>, <COURSE_NAME>, <DATE> before submission.*
"""

    REPORT.write_text(body, encoding="utf-8")
    copy = REPORTS / "report.md"
    copy.write_text(body, encoding="utf-8")
    print(f"Report written: {REPORT}")
    print(f"Copy: {copy}")
    print("\n[COMPLETE] generate_report.py")


if __name__ == "__main__":
    main()
