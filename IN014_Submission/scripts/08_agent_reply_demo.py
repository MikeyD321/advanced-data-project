#!/usr/bin/env python3
"""Demo agent reply generation (template-based; optional FLAN-T5 if transformers installed)."""
from __future__ import annotations

import json
from textwrap import shorten

import pandas as pd

from _paths import CLEANED, METRICS, SAMPLES, ensure_output_dirs

N_SAMPLES = 6


def template_reply(row: pd.Series) -> str:
    issue = row.get("issue_type", "your request")
    priority = row.get("priority", "medium")
    return (
        f"Thank you for reaching out about {issue}. We have logged your ticket "
        f"as {priority} priority and a specialist will follow up shortly."
    )


def try_flan(prompt: str) -> str | None:
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError:
        return None
    name = "google/flan-t5-small"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=400).to(device)
    out = model.generate(**inputs, max_new_tokens=80)
    return tok.decode(out[0], skip_special_tokens=True)


def main() -> None:
    ensure_output_dirs()
    df = pd.read_csv(CLEANED / "tickets_enriched.csv", low_memory=False)
    df = df[df["agent_first_reply"].notna() & df["initial_message"].notna()]
    sample = df.sample(min(N_SAMPLES, len(df)), random_state=42)

    examples = []
    mode = "template"
    for _, row in sample.iterrows():
        prompt = (
            f"Write a support agent reply. Priority: {row.get('priority')}. "
            f"Issue: {row.get('issue_type')}. Customer: {shorten(str(row['initial_message']), 200)}"
        )
        generated = try_flan(prompt) if len(examples) == 0 else None  # try model once only
        if generated is None:
            generated = template_reply(row)
        else:
            mode = "flan-t5-small (1 sample)"
        examples.append({
            "ticket_id": row["ticket_id"],
            "customer_message": shorten(str(row["initial_message"]), 150),
            "actual_reply": shorten(str(row["agent_first_reply"]), 150),
            "generated_reply": generated,
        })

    out = SAMPLES / "agent_reply_examples.txt"
    with open(out, "w") as f:
        f.write(f"Agent reply demo — mode: {mode}\n\n")
        for i, ex in enumerate(examples, 1):
            f.write(f"--- Example {i} ---\n")
            f.write(f"Customer: {ex['customer_message']}\n")
            f.write(f"Actual:   {ex['actual_reply']}\n")
            f.write(f"Generated:{ex['generated_reply']}\n\n")

    with open(METRICS / "08_agent_reply_demo.json", "w") as f:
        json.dump({"mode": mode, "n_examples": len(examples), "file": str(out.relative_to(out.parent.parent.parent))}, f, indent=2)

    print(f"Wrote {out} ({len(examples)} examples, mode={mode})")
    print("\n[COMPLETE] 08_agent_reply_demo.py — teammates can extend with fine-tuning.")


if __name__ == "__main__":
    main()
