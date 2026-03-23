from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from spider_metrics import enrich_eval_frame, analyze_eval_frame

RAW_DIR = Path("spider_testing/raw")
ENRICHED_DIR = Path("spider_testing/enriched")


def enrich_file(csv_path: str) -> None:
    input_path = Path(csv_path)
    if not input_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    ENRICHED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    enriched = enrich_eval_frame(df, dialect="sqlite")
    payload = analyze_eval_frame(df, run_name=input_path.stem, dialect="sqlite")

    enriched_csv = ENRICHED_DIR / f"{input_path.stem}_enriched.csv"
    summary_csv = ENRICHED_DIR / f"{input_path.stem}_summary.csv"

    enriched.to_csv(enriched_csv, index=False)
    payload["summary_df"].to_csv(summary_csv, index=False)

    print(f"\nSaved enriched CSV : {enriched_csv}")
    print(f"Saved summary CSV  : {summary_csv}")
    print("\nSummary:")
    print(payload["summary_df"].to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("python enrich_spider_csv.py <csv1> <csv2> ...")
        sys.exit(1)

    for csv_file in sys.argv[1:]:
        enrich_file(csv_file)