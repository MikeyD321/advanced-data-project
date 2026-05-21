# IT Support Ticket Analytics — IN014

**Team:** Miquel Raurich, Bruno Mazzoli, Pablo Bagan, Hugo  
**Course:** IN014 — Advanced Data Processing and Analysis

100k IT support tickets: star-schema tables, dashboard, stats on resolution time, and baseline models for CSAT, resolution hours, and sentiment.

**Start here:** `START_HERE.md`

## Key results

- **100,000** tickets; **60,113** resolved (~40% still open).
- **70,059** CSAT surveys (**70.1%**); use `csat_response`, not `csat_score == 0` alone.
- **Priority** drives resolution time (ANOVA p < 0.001); urgent ~**26.7 h** mean vs low ~**55.6 h**.
- **SLA plan** not significant for resolution time here (p = **0.44**).
- **CSAT model:** RMSE **1.14**, R-squared **0.089**.
- **Resolution model:** RMSE **50 h**, R-squared **-0.05** (baseline only).
- **Sentiment:** accuracy **23.8%**, macro F1 **0.21**.

## Known limitations

- Only ~**1,046** ticket `customer_id`s appear in the customer file (~90% of ticket customers have no master row).
- **17,634** regions set to `Unknown` after country imputation.
- SLA charts use simple hour thresholds, not business calendars.
- Synthetic labels limit how realistic the ML scores look.

## Layout

```
├── START_HERE.md
├── report.pdf / report.md
├── run_all.sh
├── scripts/
├── data/cleaned/          # included — pipeline input
└── outputs/charts|metrics|models|samples/
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
bash run_all.sh
```

Optional full ETL: put course CSVs in `data/raw/` and run `bash run_all.sh` again.

## PDF

`report.pdf` is already in the folder. To regenerate: `pandoc report.md -o report.pdf --toc`
