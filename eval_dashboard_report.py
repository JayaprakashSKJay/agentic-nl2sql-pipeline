import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def load_eval_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Normalize expected columns if missing
    if "exec_match" not in df.columns:
        raise ValueError(f"'exec_match' column missing in {csv_path}")

    if "latency_ms" not in df.columns:
        df["latency_ms"] = None

    if "pred_sql" not in df.columns:
        df["pred_sql"] = ""

    if "gold_sql" not in df.columns:
        df["gold_sql"] = ""

    if "message" not in df.columns:
        df["message"] = ""

    return df


def compute_metrics(df: pd.DataFrame, run_name: str) -> dict:
    total = len(df)
    correct = int(df["exec_match"].fillna(False).sum())
    execution_accuracy = correct / total if total else 0.0

    exact_match = (
        (df["pred_sql"].fillna("").str.strip() == df["gold_sql"].fillna("").str.strip())
    ).mean() if total else 0.0

    pred_exec_error = df["message"].fillna("").str.contains("Pred execution error", case=False).sum()
    sql_validity_rate = 1 - (pred_exec_error / total) if total else 0.0

    avg_latency = df["latency_ms"].dropna().mean() if total else 0.0
    median_latency = df["latency_ms"].dropna().median() if total else 0.0
    p95_latency = df["latency_ms"].dropna().quantile(0.95) if total else 0.0

    result_mismatch = df["message"].fillna("").str.contains("Result mismatch", case=False).sum()
    other_errors = total - correct - result_mismatch

    return {
        "run_name": run_name,
        "examples": total,
        "correct": correct,
        "execution_accuracy": round(execution_accuracy, 4),
        "exact_match_accuracy": round(float(exact_match), 4),
        "sql_validity_rate": round(float(sql_validity_rate), 4),
        "avg_latency_ms": round(float(avg_latency), 2) if pd.notna(avg_latency) else None,
        "median_latency_ms": round(float(median_latency), 2) if pd.notna(median_latency) else None,
        "p95_latency_ms": round(float(p95_latency), 2) if pd.notna(p95_latency) else None,
        "result_mismatch_count": int(result_mismatch),
        "execution_error_count": int(pred_exec_error),
        "other_error_count": int(other_errors if other_errors > 0 else 0),
    }


def make_accuracy_chart(summary_df: pd.DataFrame, out_dir: Path):
    plt.figure(figsize=(8, 5))
    plt.bar(summary_df["run_name"], summary_df["execution_accuracy"])
    plt.ylabel("Execution Accuracy")
    plt.ylim(0, 1)
    plt.title("Execution Accuracy by Run")
    plt.tight_layout()
    plt.savefig(out_dir / "execution_accuracy.png")
    plt.close()


def make_latency_chart(summary_df: pd.DataFrame, out_dir: Path):
    plt.figure(figsize=(8, 5))
    plt.bar(summary_df["run_name"], summary_df["avg_latency_ms"])
    plt.ylabel("Average Latency (ms)")
    plt.title("Average Latency by Run")
    plt.tight_layout()
    plt.savefig(out_dir / "avg_latency.png")
    plt.close()


def make_outcome_chart(df: pd.DataFrame, run_name: str, out_dir: Path):
    correct = int(df["exec_match"].fillna(False).sum())
    mismatch = int(df["message"].fillna("").str.contains("Result mismatch", case=False).sum())
    exec_error = int(df["message"].fillna("").str.contains("Pred execution error", case=False).sum())
    other = len(df) - correct - mismatch - exec_error

    labels = ["Correct", "Mismatch", "Exec Error", "Other"]
    values = [correct, mismatch, exec_error, max(other, 0)]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values)
    plt.title(f"Outcome Distribution: {run_name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"{run_name}_outcomes.png")
    plt.close()


def generate_dashboard(csv_paths: list[str], out_dir: str = "eval_dashboard_output"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summaries = []

    for csv_path in csv_paths:
        run_name = Path(csv_path).stem
        df = load_eval_csv(csv_path)
        metrics = compute_metrics(df, run_name)
        summaries.append(metrics)

        # save per-run outcome chart
        make_outcome_chart(df, run_name, out_path)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_path / "summary_metrics.csv", index=False)

    make_accuracy_chart(summary_df, out_path)
    make_latency_chart(summary_df, out_path)

    print("\n=== Evaluation Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved outputs to: {out_path.resolve()}")


if __name__ == "__main__":
    # Example usage:
    # python eval_dashboard_report.py spider_eval_dev_10.csv spider_eval_dev_50.csv
    import sys

    if len(sys.argv) < 2:
        print("Usage: python eval_dashboard_report.py <csv1> <csv2> ...")
        raise SystemExit(1)

    generate_dashboard(sys.argv[1:])