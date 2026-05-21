#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

echo "=== IN014 Pipeline ==="

RAW_TICKETS="data/raw/it_support_tickets.csv"
RAW_CUSTOMERS="data/raw/customer_info.csv"

if [[ -f "$RAW_TICKETS" && -f "$RAW_CUSTOMERS" ]]; then
  echo ">>> Raw CSVs found — running full ETL (01–02)"
  for s in scripts/01_clean_data.py scripts/02_fix_quality.py; do
    echo ">>> $s"
    "$PY" "$s"
    echo
  done
else
  echo ">>> No raw CSVs in data/raw/ — using existing data/cleaned/"
  echo "    (Drop course files there and rerun to rebuild from scratch.)"
  echo
fi

for s in \
  scripts/03_dashboard.py \
  scripts/04_time_analysis.py \
  scripts/05_csat_model.py \
  scripts/06_resolution_prediction.py \
  scripts/07_sentiment_classifier.py \
  scripts/08_agent_reply_demo.py \
  scripts/generate_report.py
do
  echo ">>> $s"
  "$PY" "$s"
  echo
done
echo "Done. See report.pdf / report.md and outputs/charts/dashboard.html"
