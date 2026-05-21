# Future work — IN014 project

Concrete extensions for the team. Update **Owner** and **Status** as you go.

| # | Task | Effort | Owner | Status |
|---|------|--------|-------|--------|
| 1 | **Streamlit dashboard** — filters for region, priority, date; wraps `03_dashboard.py` charts | 4–6 h | TBD | Not started |
| 2 | **PDF / Word export** — final formatting, page numbers, cover page | 1–2 h | TBD | Not started |
| 3 | **Refresh metrics after data change** — rerun `run_all.sh`, diff `outputs/metrics/*.json` | 1 h | TBD | Not started |
| 4 | **XGBoost + tuning** for CSAT and resolution models; compare to Random Forest | 6–8 h | TBD | Not started |
| 5 | **Better resolution features** — agent reply length, reopen flag, segment interactions | 4 h | TBD | Not started |
| 6 | **BERT / sentence-transformer sentiment** — replace logistic TF-IDF baseline | 8–12 h | TBD | Not started |
| 7 | **LLM fine-tune** for `agent_first_reply` (LoRA on small model); replace template demo | 12–20 h | TBD | Not started |
| 8 | **Fairness audit** — CSAT and resolution error by `customer_segment` and `region` | 4–6 h | TBD | Not started |
| 9 | **Business-hour SLA** — exclude weekends/nights in compliance metric | 4 h | TBD | Not started |
| 10 | **Databricks / warehouse port** — same star schema in Delta + scheduled jobs | 10+ h | TBD | Not started |
| 11 | **Customer coverage fix** — build fallback `dim_customer` from ticket-level fields for orphan IDs | 3–4 h | TBD | Not started |
| 12 | **Time series module** — monthly ticket rate forecasting by product_area | 6 h | TBD | Not started |

## Quick wins (do first)

- Streamlit dashboard (#1)  
- PDF export (#2)  
## Medium priority

- XGBoost (#4), resolution features (#5), business-hour SLA (#9)  

## Advanced / portfolio pieces

- BERT sentiment (#6), LLM replies (#7), fairness (#8), warehouse (#10)  

## Notes

- Keep `outputs/metrics/*.json` as the source of truth for report numbers.  
- Any model change should update Section 5 and the reflection paragraph for whoever owns that script.
