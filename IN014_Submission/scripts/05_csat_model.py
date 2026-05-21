#!/usr/bin/env python3
"""Predict CSAT score from text + metadata (survey respondents only)."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from _paths import CLEANED, FIGURES, METRICS, MODELS, ensure_output_dirs

MAX_ROWS = 40_000
SEED = 42


def main() -> None:
    ensure_output_dirs()
    df = pd.read_csv(CLEANED / "tickets_enriched.csv", low_memory=False)
    train = df[df["csat_response"] == 1].copy() if "csat_response" in df.columns else df[df["csat_score"] > 0].copy()
    if len(train) < 200:
        print("[ERROR] Not enough CSAT responses.")
        return

    if len(train) > MAX_ROWS:
        train = train.sample(MAX_ROWS, random_state=SEED)

    train["initial_message"] = train["initial_message"].fillna("")
    cats = [c for c in ("priority", "channel", "product_area", "region", "customer_segment") if c in train.columns]
    y = train["csat_score"].astype(float)

    X_tr, X_te, y_tr, y_te = train_test_split(train, y, test_size=0.2, random_state=SEED)
    tfidf = TfidfVectorizer(max_features=2500, ngram_range=(1, 2), min_df=5)
    Xt_tr = tfidf.fit_transform(X_tr["initial_message"])
    Xt_te = tfidf.transform(X_te["initial_message"])

    if cats:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        X_train = hstack([Xt_tr, ohe.fit_transform(X_tr[cats])])
        X_test = hstack([Xt_te, ohe.transform(X_te[cats])])
        names = list(tfidf.get_feature_names_out()) + list(ohe.get_feature_names_out(cats))
    else:
        X_train, X_test, names = Xt_tr, Xt_te, list(tfidf.get_feature_names_out())

    model = RandomForestRegressor(n_estimators=120, max_depth=14, random_state=SEED, n_jobs=-1)
    model.fit(X_train, y_tr)
    pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    r2 = float(r2_score(y_te, pred))
    print(f"CSAT model — RMSE: {rmse:.3f}, R²: {r2:.3f}")

    imp = model.feature_importances_
    top = np.argsort(imp)[-20:][::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(20), imp[top][::-1])
    ax.set_yticks(range(20))
    ax.set_yticklabels([names[i] if i < len(names) else f"f{i}" for i in top][::-1], fontsize=8)
    ax.set_title("CSAT model — top features")
    fig.tight_layout()
    fig_path = FIGURES / "csat_feature_importance.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)

    import joblib
    joblib.dump({"model": model, "tfidf": tfidf, "cats": cats}, MODELS / "csat_model.joblib")

    with open(METRICS / "05_csat_model.json", "w") as f:
        json.dump({
            "model": "RandomForestRegressor",
            "n_train": len(y_tr), "n_test": len(y_te),
            "rmse": rmse, "r2": r2, "figure": fig_path.name,
        }, f, indent=2)
    print("\n[COMPLETE] 05_csat_model.py")


if __name__ == "__main__":
    main()
