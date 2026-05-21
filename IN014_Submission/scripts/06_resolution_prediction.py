#!/usr/bin/env python3
"""Predict resolution_time_hours from text + metadata (resolved tickets)."""
from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from _paths import CLEANED, FIGURES, METRICS, MODELS, ensure_output_dirs

MAX_ROWS = 35_000
SEED = 42


def main() -> None:
    ensure_output_dirs()
    df = pd.read_csv(CLEANED / "tickets_enriched.csv", low_memory=False)
    df["resolution_time_hours"] = pd.to_numeric(df["resolution_time_hours"], errors="coerce")
    df = df[df["resolution_time_hours"].notna()]
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=SEED)

    df["text"] = (df["initial_message"].fillna("") + " " + df.get("agent_first_reply", pd.Series("", index=df.index)).fillna("")).str.strip()
    cats = [c for c in ("priority", "channel", "product_area", "sla_plan", "region") if c in df.columns]
    y = np.log1p(df["resolution_time_hours"])

    tr, te = train_test_split(df, test_size=0.2, random_state=SEED)
    tfidf = TfidfVectorizer(max_features=2000, min_df=5, ngram_range=(1, 2))
    Xtr = tfidf.fit_transform(tr["text"])
    Xte = tfidf.transform(te["text"])
    if cats:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        X_train = hstack([Xtr, ohe.fit_transform(tr[cats])])
        X_test = hstack([Xte, ohe.transform(te[cats])])
    else:
        ohe = None
        X_train, X_test = Xtr, Xte

    model = RandomForestRegressor(n_estimators=100, max_depth=16, random_state=SEED, n_jobs=-1)
    model.fit(X_train, y.loc[tr.index])
    pred = np.expm1(model.predict(X_test))
    actual = np.expm1(y.loc[te.index])

    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    mae = float(mean_absolute_error(actual, pred))
    r2 = float(r2_score(actual, pred))
    print(f"Resolution model — RMSE: {rmse:.1f}h, MAE: {mae:.1f}h, R²: {r2:.3f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, pred, alpha=0.2, s=6)
    lim = max(actual.max(), pred.max())
    ax.plot([0, lim], [0, lim], "r--")
    ax.set_xlabel("Actual hours")
    ax.set_ylabel("Predicted hours")
    ax.set_title("Resolution time: actual vs predicted")
    fig.tight_layout()
    fig_path = FIGURES / "resolution_pred_scatter.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)

    joblib.dump({"model": model, "tfidf": tfidf, "ohe": ohe, "cats": cats}, MODELS / "resolution_model.joblib")

    with open(METRICS / "06_resolution_prediction.json", "w") as f:
        json.dump({"rmse_hours": rmse, "mae_hours": mae, "r2": r2, "figure": fig_path.name}, f, indent=2)
    print("\n[COMPLETE] 06_resolution_prediction.py")


if __name__ == "__main__":
    main()
