import pandas as pd
import sqlglot
from sqlglot import exp


CSV_PATH = "spider_eval_dev_100.csv"


def normalize_sql(sql: str, dialect: str = "sqlite") -> str:
    """
    Parse SQL and normalize it into a canonical form.
    This is an approximation of Spider-style exact match, not the official evaluator.
    """
    sql = (sql or "").strip()
    if not sql:
        return ""

    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception:
        return ""

    # Remove harmless LIMIT 200 that your generator appends during evaluation
    if isinstance(tree, exp.Select):
        limit_expr = tree.args.get("limit")
        if limit_expr and str(limit_expr.expression).strip() == "200":
            tree.set("limit", None)

    # Normalize identifier case
    for node in tree.find_all(exp.Identifier):
        node.set("this", node.this.lower())

    return tree.sql(dialect=dialect, pretty=False)


def exact_match(pred_sql: str, gold_sql: str, dialect: str = "sqlite") -> bool:
    pred_norm = normalize_sql(pred_sql, dialect=dialect)
    gold_norm = normalize_sql(gold_sql, dialect=dialect)
    return pred_norm == gold_norm and pred_norm != ""


def main():
    df = pd.read_csv(CSV_PATH)

    df["pred_sql_norm"] = df["pred_sql"].fillna("").apply(normalize_sql)
    df["gold_sql_norm"] = df["gold_sql"].fillna("").apply(normalize_sql)
    df["exact_match"] = df.apply(
        lambda r: exact_match(r["pred_sql"], r["gold_sql"]), axis=1
    )

    total = len(df)
    exact = int(df["exact_match"].sum())
    exact_acc = exact / total if total else 0.0

    print("\n=== Exact Match Summary ===")
    print(f"Total queries      : {total}")
    print(f"Exact matches      : {exact}")
    print(f"Exact match acc    : {exact_acc:.4f}")

    out_csv = CSV_PATH.replace(".csv", "_with_exact_match.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    print("\nSample mismatches:")
    mismatches = df[~df["exact_match"]][
        ["idx", "db_id", "question", "pred_sql", "gold_sql", "pred_sql_norm", "gold_sql_norm"]
    ].head(10)
    print(mismatches.to_string(index=False))


if __name__ == "__main__":
    main()