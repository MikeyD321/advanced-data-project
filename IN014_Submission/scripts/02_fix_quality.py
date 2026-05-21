#!/usr/bin/env python3
"""Fix data quality: dedupe dim_customer, csat_response, region imputation."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from _paths import CLEANED, METRICS, ROOT, ensure_output_dirs

COUNTRY_TO_REGION = {
    "germany": "EU", "austria": "EU", "switzerland": "EU", "france": "EU",
    "italy": "EU", "spain": "EU", "netherlands": "EU", "belgium": "EU",
    "sweden": "EU", "poland": "EU", "united kingdom": "EU", "uk": "EU",
    "brazil": "LATAM", "argentina": "LATAM", "chile": "LATAM", "mexico": "LATAM", "colombia": "LATAM",
    "india": "APAC", "japan": "APAC", "china": "APAC", "singapore": "APAC",
    "australia": "APAC", "south korea": "APAC",
    "uae": "MEA", "united arab emirates": "MEA", "saudi arabia": "MEA",
    "south africa": "MEA", "egypt": "MEA",
    "usa": "NA", "united states": "NA", "canada": "NA",
}


def map_country(country) -> str:
    if pd.isna(country):
        return "Unknown"
    return COUNTRY_TO_REGION.get(str(country).strip().lower(), "Unknown")


def region_needs_fill(series: pd.Series) -> pd.Series:
    norm = series.astype(str).str.strip().str.lower()
    return series.isna() | norm.isin(["", "nan", "unknown", "none", "nat"])


def impute_regions(df: pd.DataFrame, lookup: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    out["customer_id"] = out["customer_id"].astype(str)
    before = region_needs_fill(out["region"])
    n_before = int(before.sum())

    if "country" not in out.columns:
        merged = out.merge(lookup, on="customer_id", how="left")
    else:
        merged = out.merge(
            lookup.rename(columns={"country": "country_dim"}),
            on="customer_id",
            how="left",
        )
        merged["country"] = merged["country"].fillna(merged.get("country_dim"))
        merged = merged.drop(columns=["country_dim"], errors="ignore")
    needs = region_needs_fill(merged["region"])
    mapped = merged["country"].map(map_country)

    imputed_from_country = needs & (mapped != "Unknown")
    merged.loc[imputed_from_country, "region"] = mapped[imputed_from_country]

    still = region_needs_fill(merged["region"])
    n_unknown = int(still.sum())
    merged.loc[still, "region"] = "Unknown"

    drop_cols = [c for c in ("country", "country_dim") if c in merged.columns and c not in out.columns]
    merged = merged.drop(columns=drop_cols, errors="ignore")
    stats = {
        "missing_before": n_before,
        "imputed_from_country": int(imputed_from_country.sum()),
        "set_unknown": n_unknown,
        "missing_after": int(region_needs_fill(merged["region"]).sum()),
    }
    return merged, stats


def refresh_metrics(enriched: pd.DataFrame) -> None:
    n = len(enriched)
    miss = enriched.isnull().sum()
    pd.DataFrame({
        "column": miss.index,
        "missing_count": miss.values,
        "missing_percentage": (miss.values / n * 100).round(4),
    }).sort_values("missing_count", ascending=False).to_csv(
        METRICS / "missing_values_summary.csv", index=False
    )
    num = enriched.select_dtypes(include=[np.number])
    if not num.empty:
        num.describe().T.reset_index().rename(columns={"index": "column"}).to_csv(
            METRICS / "descriptive_statistics.csv", index=False
        )


def main() -> None:
    ensure_output_dirs()
    dim_path = CLEANED / "dim_customer.csv"
    fact_path = CLEANED / "fact_tickets.csv"
    enr_path = CLEANED / "tickets_enriched.csv"
    for p in (dim_path, fact_path, enr_path):
        if not p.exists():
            print(f"[ERROR] Missing {p}. Run 01_clean_data.py first.")
            sys.exit(1)

    dim = pd.read_csv(dim_path, low_memory=False)
    dim["customer_id"] = dim["customer_id"].astype(str)
    rows_before = len(dim)
    dup_mask = dim.duplicated(subset="customer_id", keep=False)
    dup_ids = dim.loc[dup_mask, "customer_id"].unique().tolist()
    dim = dim.drop_duplicates(subset="customer_id", keep="first").reset_index(drop=True)
    rows_removed = rows_before - len(dim)

    if rows_removed:
        label = dup_ids[0] if len(dup_ids) == 1 else f"{dup_ids[0]} (+{len(dup_ids)-1} more)"
        print(f"Removed {rows_removed} duplicate customer rows for customer_id: {label}")
    else:
        print("Removed 0 duplicate customer rows (none found).")

    fact = pd.read_csv(fact_path, low_memory=False)
    enriched = pd.read_csv(enr_path, low_memory=False)
    lookup = dim[["customer_id", "country"]].drop_duplicates("customer_id")

    for frame in (fact, enriched):
        frame["csat_score"] = pd.to_numeric(frame["csat_score"], errors="coerce").fillna(0)
    fact["csat_response"] = (fact["csat_score"] > 0).astype(int)
    enriched["csat_response"] = (enriched["csat_score"] > 0).astype(int)
    resp = int(fact["csat_response"].sum())
    tot = len(fact)
    print(
        f"Added csat_response column — {resp:,} surveys responded ({resp/tot*100:.1f}%), "
        f"{tot-resp:,} no response ({(tot-resp)/tot*100:.1f}%)"
    )

    print("fact_tickets region imputation:")
    fact, fact_st = impute_regions(fact, lookup)
    print(
        f"Imputed {fact_st['imputed_from_country']:,} missing regions from country. "
        f"{fact_st['set_unknown']:,} left as Unknown."
    )

    print("tickets_enriched region imputation:")
    enriched, enr_st = impute_regions(enriched, lookup)
    print(
        f"Imputed {enr_st['imputed_from_country']:,} missing regions from country. "
        f"{enr_st['set_unknown']:,} left as Unknown."
    )

    status = enriched["status"].astype(str).str.lower()
    empty = enriched["resolution_summary"].isna() | enriched["resolution_summary"].astype(str).str.strip().eq("")
    fill_mask = status.isin(["resolved", "closed_no_action"]) & empty
    n_filled = int(fill_mask.sum())
    enriched.loc[fill_mask, "resolution_summary"] = "No summary provided"
    print(f"Filled {n_filled:,} resolution_summary gaps for resolved/closed tickets.")

    dim.to_csv(dim_path, index=False)
    fact.to_csv(fact_path, index=False)
    enriched.to_csv(enr_path, index=False)
    refresh_metrics(enriched)

    summary = {
        "duplicate_customers_removed": rows_removed,
        "regions_imputed_fact": fact_st["imputed_from_country"],
        "regions_unknown_fact": fact_st["set_unknown"],
        "resolution_summary_filled": n_filled,
        "csat_responded": resp,
    }
    with open(METRICS / "02_fix_quality.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Data Quality Fixes Complete ===")
    print(f"Duplicate customers removed:        {rows_removed:,}")
    print(f"Missing regions imputed (fact):     {fact_st['imputed_from_country']:,}")
    print(f"Missing regions left as Unknown:    {fact_st['set_unknown']:,}")
    print(f"csat_response flags added:          {tot:,} rows")
    print(f"resolution_summary gaps filled:     {n_filled:,}")
    print("\n[COMPLETE] 02_fix_quality.py")


if __name__ == "__main__":
    main()
