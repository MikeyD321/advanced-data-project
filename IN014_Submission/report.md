# IN014 — Advanced Data Processing and Analysis

## Title Page

| | |
|---|---|
| **Project** | IT Support Ticket Analytics |
| **Course** | IN014 — Advanced Data Processing and Analysis |
| **Team** | Miquel Raurich, Bruno Mazzoli, Pablo Bagan, Hugo |
| **Date** | 2026-05-21 |
| **Repository** | https://github.com/miquelraurich/IN014_project |

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Data governance](#2-data-governance)
3. [Data stack](#3-data-stack)
4. [Data model](#4-data-model)
5. [Findings and results](#5-findings-and-results)
6. [Reflection](#6-reflection)
7. [Appendix](#appendix)

## Table of figures

| Figure | File | Section |
|--------|------|---------|
| 1 | `outputs/charts/volume_by_priority.png` | 5.2 |
| 2 | `outputs/charts/resolution_time_dist.png` | 5.2 |
| 3 | `outputs/charts/sla_by_segment.png` | 5.2 |
| 4 | `outputs/charts/dashboard.html` (interactive) | 5.2 |
| 5 | `outputs/charts/resolution_time_boxplots.png` | 5.3 |
| 6 | `outputs/charts/csat_feature_importance.png` | 5.4 |
| 7 | `outputs/charts/resolution_pred_scatter.png` | 5.5 |
| 8 | `outputs/charts/sentiment_confusion_matrix.png` | 5.6 |

## Table of tables

| Table | Description | Section |
|-------|-------------|---------|
| 1 | Data governance summary | 2 |
| 2 | Technology stack | 3 |
| 3 | Star schema tables | 4 |
| 4 | Mean resolution time by priority | 5.3 |
| 5 | ANOVA results for resolution time | 5.3 |
| 6 | Model performance summary | 5.4–5.6 |

---

## 1. Introduction

This project works with **100,000 synthetic IT support tickets** and a separate customer master file. Each ticket records when it was opened, how the customer contacted support, what went wrong, how long it took to close, and—when available—a satisfaction score and short text from the customer and agent.

**Where the data comes from:** Course CSV files (`it_support_tickets.csv`, `customer_info.csv`). This handoff includes the cleaned tables in `data/cleaned/`; raw files are optional if you only rerun charts and models.

**Business problem:** Support managers need answers to practical questions: Which channels and product areas generate the most load? Do urgent tickets actually close faster? Can we estimate CSAT or resolution time from the first message and ticket metadata? We built a small pipeline—clean tables, a dashboard, statistical checks, and baseline models—to answer those questions.

---

## 2. Data governance

| Topic | Approach |
|-------|----------|
| **Ownership** | Synthetic course data; treat `initial_message` and `agent_first_reply` as potentially sensitive |
| **Quality rules** | `resolution_time_hours` missing = ticket still open; `csat_score` = 0 means no survey (not “worst score”) |
| **Duplicates** | Exact duplicate ticket rows removed in cleaning; one row per `customer_id` in `dim_customer` |
| **Lineage** | `data/raw` → `01_clean_data.py` → `02_fix_quality.py` → `data/cleaned` → scripts 03–08 → `outputs/` |

**Important data caveat:** Only **1,046** of roughly **9,999** ticket-level `customer_id` values appear in `customer_info.csv`. Enrichment and country-based region fills only apply where a customer row exists; the rest stay sparse or `Unknown`. We call this out in Section 5 instead of pretending the customer file covers everyone.

---

## 3. Data stack

We ran everything **locally in Python** on a laptop. At 100k rows, in-memory pandas and scikit-learn were enough; we did not need a cloud warehouse for this assignment.

| Layer | Tool |
|-------|------|
| ETL | pandas, pathlib |
| Storage | CSV in `data/cleaned/` |
| Statistics / ML | scipy, scikit-learn |
| Charts | matplotlib, seaborn, plotly (`dashboard.html`) |
| Orchestration | `run_all.sh` |

The same star schema could later sit in BigQuery or Snowflake with Airflow jobs mirroring `01` and `02`. We chose local tooling so the full path from raw CSV to report figures is easy to reproduce for grading.

---

## 4. Data model

**Cleaning (`01_clean_data.py`):** Validate required columns, parse `created_at`, coerce binary fields, left-join customers into `tickets_enriched`, export fact and dimension tables.

**Quality fixes (`02_fix_quality.py`):** Deduplicate `dim_customer`, add `csat_response` (1 if `csat_score` > 0), impute missing `region` from `country` where possible.

| Table | Role |
|-------|------|
| `fact_tickets` | One row per ticket; core measures and dimensions |
| `dim_customer` | Customer attributes (1,047 unique IDs) |
| `dim_date` | Calendar from min/max ticket dates |
| `dim_channel` | Five support channels |
| `dim_product_area` | Seven product areas |
| `tickets_enriched` | Wide table for modeling |

```
data/raw  →  01_clean_data  →  02_fix_quality  →  data/cleaned
                                              →  03–08 (charts & models)
                                              →  report.md
```

---

## 5. Findings and results

### 5.1 Data quality

| Metric | Value |
|--------|------:|
| Duplicate customer rows removed | 0 |
| Regions imputed from country (fact) | 2,363 |
| Regions set to `Unknown` (unmappable / no match) | 17,634 |
| Tickets with CSAT survey (`csat_response` = 1) | 70,059 (70.1%) |

About **39.9%** of tickets have no `resolution_time_hours` (still open or not closed in the extract). Roughly **30%** have no CSAT response.

### 5.2 Dashboard

![Ticket volume by priority over time](outputs/charts/volume_by_priority.png)

*Figure 1: Monthly ticket counts stacked by priority.*

![Resolution time distribution](outputs/charts/resolution_time_dist.png)

*Figure 2: Distribution of resolution hours for closed tickets.*

![SLA compliance by customer segment](outputs/charts/sla_by_segment.png)

*Figure 3: Share of tickets meeting a simple hour-based SLA rule, by segment.*

The interactive dashboard (`outputs/charts/dashboard.html`) adds channel mix, sentiment breakdown, and CSAT histograms in one view.

### 5.3 Resolution time analysis

Among **60,113** resolved tickets:

| Statistic | Hours |
|-----------|------:|
| Mean | 45.0 |
| Median | 29.9 |

**Mean resolution time by priority**

| Priority | Mean (h) | Median (h) | n |
|----------|---------:|-----------:|--:|
| urgent | 26.7 | 4.9 | 3,043 |
| high | 31.6 | 14.7 | 11,882 |
| medium | 43.3 | 29.8 | 21,358 |
| low | 55.6 | 45.5 | 23,830 |

Urgent and high-priority tickets close faster on average; low-priority tickets sit much longer—consistent with how queues are usually staffed.

**ANOVA on resolution time**

| Factor | F | p-value | Significant at 5% level? |
|--------|--:|--------:|:--:|
| priority | 848.02 | < 0.001 | Yes |
| sla_plan | 0.82 | 0.442 | No |
| region | 2.17 | 0.070 | No |

So **priority drives differences in resolution time** in this dataset; **SLA tier and region do not** show significant separation at the 5% significance level (region is borderline at p about 0.07).

![Resolution time by priority, SLA, and region](outputs/charts/resolution_time_boxplots.png)

*Figure 5: Boxplots of resolution hours across priority, SLA plan, and region.*

### 5.4 CSAT prediction

We trained a **Random Forest** on tickets with `csat_response` = 1, using TF-IDF on `initial_message` plus categorical fields (40,000 train / 10,000 test sample cap).

| Metric | Value |
|--------|------:|
| RMSE | 1.14 |
| R-squared | 0.089 |

The model picks up a weak signal: text and metadata alone explain under **9%** of score variance. That is plausible—many drivers of satisfaction (agent skill, follow-ups) are not in the table.

![CSAT model feature importance](outputs/charts/csat_feature_importance.png)

*Figure 6: Top features for CSAT regression.*

### 5.5 Resolution time prediction

Random Forest on resolved tickets (log-transformed target, text + metadata):

| Metric | Value |
|--------|------:|
| RMSE | 50.0 h |
| MAE | 27.0 h |
| R-squared | -0.046 |

Negative R-squared means the model does **worse than predicting the mean** on the holdout split. Resolution time is noisy and heavy-tailed; a simple bag-of-words model is not enough. We still include it to show what we tried and where a teammate could add better features or gradient boosting.

![Actual vs predicted resolution hours](outputs/charts/resolution_pred_scatter.png)

*Figure 7: Predicted vs actual resolution hours.*

### 5.6 Sentiment classification

Logistic regression + TF-IDF on `initial_message` → `customer_sentiment` (25,000 ticket sample):

| Metric | Value |
|--------|------:|
| Accuracy | 0.238 |
| Macro F1 | 0.211 |
| Weighted F1 | 0.198 |

Performance is modest—the five-way labels are hard to recover from text alone, and labels are synthetic. The confusion matrix still shows which sentiments get mixed up (e.g. negative vs very_negative).

![Sentiment confusion matrix](outputs/charts/sentiment_confusion_matrix.png)

*Figure 8: Confusion matrix for sentiment classifier.*

### 5.7 Agent reply generation (demo)

We generated **6** example replies using a **template-based** demo (`08_agent_reply_demo.py`). Examples are in `outputs/samples/agent_reply_examples.txt`. Next step could be fine-tuning a small LLM on `initial_message` → `agent_first_reply` (see `FUTURE_WORK.md`).

---

## 6. Reflection

**Miquel Raurich:** I spent most of my time on the data pipeline and star schema, not on tuning models. The surprise was how thin the customer file is relative to tickets—only about one in ten ticket `customer_id`s match `customer_info.csv`, which meant **17,634** regions ended up as `Unknown` after imputation. That changed how I read every downstream chart. Building `csat_response` as its own flag also clarified the difference between “score is zero” and “customer skipped the survey.” If I redid the project, I would add a ticket-only customer dimension so orphans are not left blank.

**Bruno Mazzoli:** I focused on whether resolution time actually differs by operational levers. Priority clearly matters—ANOVA F = **848** with p &lt; 0.001—and urgent tickets average about **26.7 hours** compared with **55.6** for low priority. What I did not expect was that **SLA plan** was not significant (p = **0.44**): gold and standard tiers look similar once tickets are resolved. That made me think SLA labels in synthetic data may not line up with resolution behavior the way real contracts would. I would like to add business-hour SLA logic next.

**Pablo Bagan:** The modeling section was humbling. CSAT regression only reached R-squared about **0.089**, and resolution time prediction was negative on the holdout set. At first that felt like failure, but it is an honest result: text n-grams plus a few categories do not capture why a ticket took 200 hours or why someone gave a 2 vs 4. The sentiment classifier at **23.8%** accuracy showed the same story. I learned to report weak baselines clearly instead of hiding them—reviewers care more about whether you understand limits than about a fake 0.95 AUC.

**Hugo:** I pulled the dashboard and presentation pieces together. The static charts and `dashboard.html` made the volume story easy to tell—especially priority mix over time and the long tail of resolution hours. Linking every number in the report back to `outputs/metrics/*.json` kept the group from arguing over rounding. For submission, I would export the report to PDF and add a short Streamlit wrapper so a reader can filter by region without rerunning notebooks. Overall the project felt like real support analytics, not just a homework CSV exercise.

---

## Appendix

### Reproduce

```bash
git clone https://github.com/miquelraurich/IN014_project
cd IN014_project
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
bash run_all.sh
# Optional: place course CSVs in data/raw/ to rebuild cleaned tables from scratch
```

### Script index

| Script | Purpose |
|--------|---------|
| `01_clean_data.py` | ETL + star schema |
| `02_fix_quality.py` | DQ fixes |
| `03_dashboard.py` | Charts + HTML dashboard |
| `04_time_analysis.py` | ANOVA + boxplots |
| `05_csat_model.py` | CSAT regression |
| `06_resolution_prediction.py` | Resolution regression |
| `07_sentiment_classifier.py` | Sentiment ML |
| `08_agent_reply_demo.py` | Reply demo |
| `generate_report.py` | Report draft generator |

### GitHub

Code and documentation: **https://github.com/miquelraurich/IN014_project**

Cleaned tables ship with this folder. Raw course CSVs are optional for a full rebuild (`data/raw/`).

---

*IN014 Final Project — IT Support Ticket Analytics. Submitted May 2026.*
