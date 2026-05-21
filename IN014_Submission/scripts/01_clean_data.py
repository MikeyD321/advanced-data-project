#!/usr/bin/env python3
"""
01_clean_data.py — Load raw IT support data, clean, enrich, and build star schema.

Run from project root:
    python scripts/01_clean_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (project root = parent of scripts/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

TICKETS_FILE = RAW_DIR / "it_support_tickets.csv"
CUSTOMERS_FILE = RAW_DIR / "customer_info.csv"

REQUIRED_TICKET_COLUMNS = [
    "ticket_id",
    "created_at",
    "customer_id",
    "customer_segment",
    "channel",
    "product_area",
    "issue_type",
    "priority",
    "status",
    "sla_plan",
    "initial_message",
    "agent_first_reply",
    "resolution_summary",
    "resolution_time_hours",
    "reopened",
    "customer_sentiment",
    "csat_score",
    "has_attachment",
    "platform",
    "region",
]

REQUIRED_CUSTOMER_COLUMNS = [
    "customer_id",
]

FACT_COLUMNS = [
    "ticket_id",
    "customer_id",
    "date_key",
    "priority",
    "status",
    "sla_plan",
    "channel",
    "product_area",
    "issue_type",
    "resolution_time_hours",
    "reopened",
    "customer_sentiment",
    "csat_score",
    "has_attachment",
    "platform",
    "region",
]


def ensure_directories() -> None:
    """Create output directories if missing."""
    for directory in (CLEANED_DIR, METRICS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Directories ready: {CLEANED_DIR.relative_to(PROJECT_ROOT)}, "
          f"{METRICS_DIR.relative_to(PROJECT_ROOT)}")


def load_csv(path: Path, label: str) -> pd.DataFrame:
    """Load a CSV or exit with a clear error."""
    if not path.exists():
        print(f"[ERROR] Missing file: {path}")
        print(f"        Place {path.name} in {RAW_DIR.relative_to(PROJECT_ROOT)}/")
        sys.exit(1)
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        print(f"[ERROR] Could not read {label} from {path}: {exc}")
        sys.exit(1)
    print(f"[OK] Loaded {label}: {len(df):,} rows, {len(df.columns)} columns")
    print(f"     Columns: {list(df.columns)}")
    return df


def validate_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    """Ensure all required columns exist."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"[ERROR] {label} is missing required column(s): {missing}")
        sys.exit(1)
    print(f"[OK] {label} has all required columns ({len(required)} checked)")


def to_binary_int(series: pd.Series) -> pd.Series:
    """Coerce values to 0/1 integers."""
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    return (numeric > 0).astype(np.int64)


def clean_tickets(tickets: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning rules to the tickets dataframe."""
    df = tickets.copy()

    # IDs as strings
    df["ticket_id"] = df["ticket_id"].astype(str)
    df["customer_id"] = df["customer_id"].astype(str)

    # Datetime
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    # Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    print(f"[OK] Dropped {dropped:,} exact duplicate ticket row(s)")

    # Numeric fields
    df["resolution_time_hours"] = pd.to_numeric(df["resolution_time_hours"], errors="coerce")
    df["csat_score"] = pd.to_numeric(df["csat_score"], errors="coerce").fillna(0).astype(np.int64)

    # Binary fields
    df["reopened"] = to_binary_int(df["reopened"])
    df["has_attachment"] = to_binary_int(df["has_attachment"])

    # Text columns: strip whitespace only (do not impute)
    for col in ("initial_message", "agent_first_reply", "resolution_summary"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    # Categorical-like columns: leave NaN; report later in missing summary
    print("[OK] Ticket cleaning complete (NaN resolution_time_hours kept for unresolved tickets)")
    return df


def clean_customers(customers: pd.DataFrame) -> pd.DataFrame:
    """Clean customer dimension source data."""
    df = customers.copy()
    df["customer_id"] = df["customer_id"].astype(str)
    df = df.drop_duplicates(subset=["customer_id"], keep="first")
    print(f"[OK] Customer table: {len(df):,} unique customer_id values")
    return df


def merge_enriched(tickets: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """Left join tickets with customer attributes (no duplicate column names)."""
    customer_cols = [c for c in customers.columns if c not in tickets.columns or c == "customer_id"]
    customers_subset = customers[customer_cols].copy()
    enriched = tickets.merge(customers_subset, on="customer_id", how="left")
    print(f"[OK] Built tickets_enriched: {len(enriched):,} rows, {len(enriched.columns)} columns")
    return enriched


def make_date_key(created_at: pd.Series) -> pd.Series:
    """YYYYMMDD integer from datetime; NaN where created_at is NaT."""
    valid = created_at.notna()
    keys = pd.Series(np.nan, index=created_at.index, dtype="float64")
    keys.loc[valid] = (
        created_at.loc[valid].dt.year * 10000
        + created_at.loc[valid].dt.month * 100
        + created_at.loc[valid].dt.day
    )
    return keys


def build_fact_tickets(tickets: pd.DataFrame) -> pd.DataFrame:
    """Build fact_tickets with date_key from created_at."""
    fact = tickets.copy()
    fact["date_key"] = make_date_key(fact["created_at"])
    # Use nullable Int64 for date_key where possible
    fact["date_key"] = fact["date_key"].astype("Int64")
    available = [c for c in FACT_COLUMNS if c in fact.columns]
    missing_fact = [c for c in FACT_COLUMNS if c not in fact.columns]
    if missing_fact:
        print(f"[WARN] fact_tickets missing columns (will omit): {missing_fact}")
    fact_out = fact[available].copy()
    print(f"[OK] Built fact_tickets: {len(fact_out):,} rows")
    return fact_out


def build_dim_date(tickets: pd.DataFrame) -> pd.DataFrame:
    """Calendar dimension from min to max created_at."""
    valid_dates = tickets["created_at"].dropna()
    if valid_dates.empty:
        print("[WARN] No valid created_at values; dim_date will be empty")
        return pd.DataFrame(columns=["date", "date_key", "year", "month", "day", "day_of_week"])

    min_date = valid_dates.min().normalize()
    max_date = valid_dates.max().normalize()
    dates = pd.date_range(min_date, max_date, freq="D")
    dim = pd.DataFrame({"date": dates})
    dim["date_key"] = (
        dim["date"].dt.year * 10000 + dim["date"].dt.month * 100 + dim["date"].dt.day
    ).astype(np.int64)
    dim["year"] = dim["date"].dt.year.astype(np.int64)
    dim["month"] = dim["date"].dt.month.astype(np.int64)
    dim["day"] = dim["date"].dt.day.astype(np.int64)
    dim["day_of_week"] = dim["date"].dt.day_name()
    print(f"[OK] Built dim_date: {len(dim):,} days ({min_date.date()} to {max_date.date()})")
    return dim


def build_dim_channel(tickets: pd.DataFrame) -> pd.DataFrame:
    """Distinct channels from tickets."""
    dim = (
        tickets[["channel"]]
        .dropna()
        .drop_duplicates()
        .sort_values("channel")
        .reset_index(drop=True)
    )
    dim = dim.rename(columns={"channel": "channel_name"})
    dim.insert(0, "channel_key", range(1, len(dim) + 1))
    print(f"[OK] Built dim_channel: {len(dim):,} distinct channels")
    return dim


def build_dim_product_area(tickets: pd.DataFrame) -> pd.DataFrame:
    """Distinct product areas from tickets."""
    dim = (
        tickets[["product_area"]]
        .dropna()
        .drop_duplicates()
        .sort_values("product_area")
        .reset_index(drop=True)
    )
    dim = dim.rename(columns={"product_area": "product_area_name"})
    dim.insert(0, "product_area_key", range(1, len(dim) + 1))
    print(f"[OK] Built dim_product_area: {len(dim):,} distinct product areas")
    return dim


def save_table(df: pd.DataFrame, path: Path) -> None:
    """Write dataframe to CSV."""
    df.to_csv(path, index=False)
    print(f"[OK] Saved {path.relative_to(PROJECT_ROOT)} ({len(df):,} rows)")


def write_missing_summary(enriched: pd.DataFrame) -> None:
    """missing_values_summary.csv in outputs/metrics/."""
    n = len(enriched)
    missing_count = enriched.isnull().sum()
    summary = pd.DataFrame({
        "column": missing_count.index,
        "missing_count": missing_count.values,
        "missing_percentage": (missing_count.values / n * 100).round(4) if n else 0,
    })
    summary = summary.sort_values("missing_count", ascending=False).reset_index(drop=True)
    path = METRICS_DIR / "missing_values_summary.csv"
    summary.to_csv(path, index=False)
    print(f"[OK] Saved {path.relative_to(PROJECT_ROOT)}")


def write_descriptive_statistics(enriched: pd.DataFrame) -> None:
    """descriptive_statistics.csv for numeric columns."""
    numeric = enriched.select_dtypes(include=[np.number])
    if numeric.empty:
        print("[WARN] No numeric columns for descriptive statistics")
        stats = pd.DataFrame()
    else:
        stats = numeric.describe().T.reset_index().rename(columns={"index": "column"})
    path = METRICS_DIR / "descriptive_statistics.csv"
    stats.to_csv(path, index=False)
    print(f"[OK] Saved {path.relative_to(PROJECT_ROOT)} ({len(stats)} numeric columns)")


def main() -> None:
    print("=" * 60)
    print("IT Support Tickets — Data cleaning pipeline")
    print(f"Project root: {PROJECT_ROOT}")
    print("=" * 60)

    ensure_directories()

    # Load raw data
    tickets = load_csv(TICKETS_FILE, "it_support_tickets")
    customers = load_csv(CUSTOMERS_FILE, "customer_info")

    validate_columns(tickets, REQUIRED_TICKET_COLUMNS, "it_support_tickets")
    validate_columns(customers, REQUIRED_CUSTOMER_COLUMNS, "customer_info")

    # Clean
    tickets = clean_tickets(tickets)
    customers = clean_customers(customers)

    # Enriched table
    enriched = merge_enriched(tickets, customers)
    save_table(enriched, CLEANED_DIR / "tickets_enriched.csv")

    # Star schema
    fact = build_fact_tickets(tickets)
    dim_customer = customers
    dim_date = build_dim_date(tickets)
    dim_channel = build_dim_channel(tickets)
    dim_product_area = build_dim_product_area(tickets)

    save_table(fact, CLEANED_DIR / "fact_tickets.csv")
    save_table(dim_customer, CLEANED_DIR / "dim_customer.csv")
    save_table(dim_date, CLEANED_DIR / "dim_date.csv")
    save_table(dim_channel, CLEANED_DIR / "dim_channel.csv")
    save_table(dim_product_area, CLEANED_DIR / "dim_product_area.csv")

    # Summaries
    print("-" * 60)
    print("Writing summary metrics...")
    write_missing_summary(enriched)
    write_descriptive_statistics(enriched)

    print("=" * 60)
    print("[COMPLETE] 01_clean_data.py finished successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
