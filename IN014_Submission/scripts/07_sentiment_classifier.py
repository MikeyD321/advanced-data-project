#!/usr/bin/env python3
"""Sentiment classification from initial_message."""
from __future__ import annotations

import json
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from _paths import CLEANED, FIGURES, METRICS, MODELS, ensure_output_dirs

MAX_ROWS = 25_000
SEED = 42


def clean_text(t: str) -> str:
    t = re.sub(r"http\S+", " ", str(t).lower())
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def main() -> None:
    ensure_output_dirs()
    df = pd.read_csv(CLEANED / "tickets_enriched.csv", low_memory=False)
    df = df.dropna(subset=["initial_message", "customer_sentiment"])
    df = df[df["initial_message"].astype(str).str.len() > 5]
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=SEED)

    X = df["initial_message"].map(clean_text)
    y = df["customer_sentiment"].astype(str)
    strat = y if y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=strat)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=3)),
        ("clf", LogisticRegression(max_iter=400, class_weight="balanced", n_jobs=-1)),
    ])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    report = classification_report(y_te, y_pred, output_dict=True)
    print(classification_report(y_te, y_pred))

    labels = sorted(y.unique())
    cm = confusion_matrix(y_te, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Sentiment classifier — confusion matrix")
    fig.tight_layout()
    fig_path = FIGURES / "sentiment_confusion_matrix.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)

    import joblib
    joblib.dump(pipe, MODELS / "sentiment_model.joblib")

    with open(METRICS / "07_sentiment_classifier.json", "w") as f:
        json.dump({
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "figure": fig_path.name,
        }, f, indent=2)
    print("\n[COMPLETE] 07_sentiment_classifier.py")


if __name__ == "__main__":
    main()
