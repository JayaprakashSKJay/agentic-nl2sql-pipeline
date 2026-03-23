from __future__ import annotations

import json
import re
from typing import Dict, Any, List, Tuple

import pandas as pd
import sqlglot
from sqlglot import exp


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _safe_json_loads(x: str) -> dict:
    try:
        if pd.isna(x):
            return {}
        if isinstance(x, dict):
            return x
        return json.loads(x)
    except Exception:
        return {}


def _substitute_params(sql: str, params: dict) -> str:
    """
    Replace :p1 style params with literal values so exact-match approximation
    becomes more realistic.
    """
    sql = (sql or "").strip()
    if not sql or not params:
        return sql

    out = sql
    for key, value in params.items():
        placeholder = f":{key}"

        if value is None:
            replacement = "NULL"
        elif isinstance(value, (int, float)):
            replacement = str(value)
        else:
            escaped = str(value).replace("'", "''")
            replacement = f"'{escaped}'"

        out = out.replace(placeholder, replacement)

    return out


def _canon_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*,\s*", ",", s)
    s = re.sub(r"\(\s*", "(", s)
    s = re.sub(r"\s*\)", ")", s)
    return s


# -----------------------------------------------------------------------------
# SQL normalization / exact match
# -----------------------------------------------------------------------------
def normalize_sql(sql: str, dialect: str = "sqlite") -> str:
    """
    Normalize SQL into a canonical form using sqlglot.
    This is an approximation of Spider exact match, not the official evaluator.
    """
    sql = (sql or "").strip()
    if not sql:
        return ""

    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception:
        return ""

    # Remove harmless LIMIT 200 appended by generator during eval
    if isinstance(tree, exp.Select):
        limit_expr = tree.args.get("limit")
        if limit_expr and getattr(limit_expr, "expression", None):
            limit_val = str(limit_expr.expression).strip()
            if limit_val == "200":
                tree.set("limit", None)

    # Normalize identifier case
    for node in tree.find_all(exp.Identifier):
        if node.this:
            node.set("this", node.this.lower())

    normalized = tree.sql(dialect=dialect, pretty=False)
    return _canon_text(normalized)


def exact_match_flag(
    pred_sql: str,
    gold_sql: str,
    pred_params: dict | None = None,
    dialect: str = "sqlite",
) -> bool:
    pred_sql_filled = _substitute_params(pred_sql, pred_params or {})
    pred_norm = normalize_sql(pred_sql_filled, dialect=dialect)
    gold_norm = normalize_sql(gold_sql, dialect=dialect)
    return bool(pred_norm and gold_norm and pred_norm == gold_norm)


# -----------------------------------------------------------------------------
# Component extraction / matching
# -----------------------------------------------------------------------------
def _extract_select(sql: str) -> str:
    m = re.search(r"select (.*?) from", (sql or "").lower(), re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_where(sql: str) -> str:
    m = re.search(r"where (.*?)(group by|order by|limit|$)", (sql or "").lower(), re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_group(sql: str) -> str:
    m = re.search(r"group by (.*?)(order by|limit|$)", (sql or "").lower(), re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_order(sql: str) -> str:
    m = re.search(r"order by (.*?)(limit|$)", (sql or "").lower(), re.DOTALL)
    return m.group(1).strip() if m else ""


def _count_joins(sql: str) -> int:
    return len(re.findall(r"\bjoin\b", (sql or "").lower()))


def _component_equal(a: str, b: str) -> bool:
    return _canon_text(a) == _canon_text(b)


# -----------------------------------------------------------------------------
# Hardness classification
# -----------------------------------------------------------------------------
def _has_nested_query(sql: str) -> bool:
    sql_l = (sql or "").lower()
    return sql_l.count("select") > 1


def _has_group_or_having(sql: str) -> bool:
    sql_l = (sql or "").lower()
    return ("group by" in sql_l) or ("having" in sql_l)


def _has_set_ops(sql: str) -> bool:
    sql_l = (sql or "").lower()
    return any(op in sql_l for op in [" union ", " intersect ", " except "])


def _has_order_limit(sql: str) -> bool:
    sql_l = (sql or "").lower()
    return ("order by" in sql_l) or ("limit" in sql_l)


def classify_spider_hardness(sql: str) -> str:
    """
    Practical Spider-inspired hardness approximation based on SQL structure.
    """
    joins = _count_joins(sql)
    nested = _has_nested_query(sql)
    group_or_having = _has_group_or_having(sql)
    set_ops = _has_set_ops(sql)
    order_limit = _has_order_limit(sql)

    score = 0

    if joins == 1:
        score += 1
    elif joins >= 2:
        score += 2

    if nested:
        score += 2

    if group_or_having:
        score += 1

    if set_ops:
        score += 2

    if order_limit:
        score += 1

    if score <= 1:
        return "easy"
    elif score <= 3:
        return "medium"
    elif score <= 5:
        return "hard"
    else:
        return "extra_hard"


# -----------------------------------------------------------------------------
# Error / outcome helpers
# -----------------------------------------------------------------------------
def classify_outcome(message: str, exec_match: bool) -> str:
    msg = (message or "").lower()

    if bool(exec_match):
        return "correct"
    if "pred execution error" in msg:
        return "execution_error"
    if "gold execution error" in msg:
        return "gold_error"
    if "result mismatch" in msg:
        return "result_mismatch"
    if "api key" in msg or "not set" in msg:
        return "provider_error"
    if "json" in msg or "parse" in msg:
        return "parse_error"
    return "other"


# -----------------------------------------------------------------------------
# Main analyzer
# -----------------------------------------------------------------------------
def enrich_eval_frame(df: pd.DataFrame, dialect: str = "sqlite") -> pd.DataFrame:
    out = df.copy()

    # Ensure expected columns
    for col, default in {
        "pred_sql": "",
        "gold_sql": "",
        "pred_params": "{}",
        "message": "",
        "latency_ms": None,
        "exec_match": False,
        "db_id": "",
        "question": "",
    }.items():
        if col not in out.columns:
            out[col] = default

    out["pred_params_obj"] = out["pred_params"].apply(_safe_json_loads)
    out["pred_sql_filled"] = out.apply(
        lambda r: _substitute_params(str(r["pred_sql"]), r["pred_params_obj"]),
        axis=1,
    )

    out["pred_sql_norm"] = out["pred_sql_filled"].fillna("").apply(
        lambda x: normalize_sql(str(x), dialect=dialect)
    )
    out["gold_sql_norm"] = out["gold_sql"].fillna("").apply(
        lambda x: normalize_sql(str(x), dialect=dialect)
    )
    out["exact_match"] = out.apply(
        lambda r: exact_match_flag(
            str(r["pred_sql"]),
            str(r["gold_sql"]),
            pred_params=r["pred_params_obj"],
            dialect=dialect,
        ),
        axis=1,
    )

    out["select_match"] = out.apply(
        lambda r: _component_equal(
            _extract_select(str(r["pred_sql_filled"])),
            _extract_select(str(r["gold_sql"])),
        ),
        axis=1,
    )
    out["where_match"] = out.apply(
        lambda r: _component_equal(
            _extract_where(str(r["pred_sql_filled"])),
            _extract_where(str(r["gold_sql"])),
        ),
        axis=1,
    )
    out["group_match"] = out.apply(
        lambda r: _component_equal(
            _extract_group(str(r["pred_sql_filled"])),
            _extract_group(str(r["gold_sql"])),
        ),
        axis=1,
    )
    out["order_match"] = out.apply(
        lambda r: _component_equal(
            _extract_order(str(r["pred_sql_filled"])),
            _extract_order(str(r["gold_sql"])),
        ),
        axis=1,
    )
    out["join_match"] = out.apply(
        lambda r: _count_joins(str(r["pred_sql_filled"])) == _count_joins(str(r["gold_sql"])),
        axis=1,
    )

    out["join_count"] = out["gold_sql"].fillna("").apply(_count_joins)
    out["complexity"] = out["gold_sql"].fillna("").apply(classify_spider_hardness)

    out["outcome_type"] = out.apply(
        lambda r: classify_outcome(str(r["message"]), bool(r["exec_match"])),
        axis=1,
    )

    return out


def analyze_eval_frame(df: pd.DataFrame, run_name: str, dialect: str = "sqlite") -> Dict[str, Any]:
    needed_cols = {
        "exact_match",
        "select_match",
        "where_match",
        "group_match",
        "order_match",
        "join_match",
        "complexity",
        "outcome_type",
    }
    if needed_cols.issubset(set(df.columns)):
        edf = df.copy()
    else:
        edf = enrich_eval_frame(df, dialect=dialect)

    total = len(edf)
    correct = int(edf["exec_match"].fillna(False).sum())
    execution_accuracy = correct / total if total else 0.0
    exact_match_accuracy = float(edf["exact_match"].mean()) if total else 0.0

    pred_exec_error = int((edf["outcome_type"] == "execution_error").sum())
    sql_validity_rate = 1 - (pred_exec_error / total) if total else 0.0

    component_scores = {
        "select_accuracy": float(edf["select_match"].mean()) if total else 0.0,
        "where_accuracy": float(edf["where_match"].mean()) if total else 0.0,
        "group_accuracy": float(edf["group_match"].mean()) if total else 0.0,
        "order_accuracy": float(edf["order_match"].mean()) if total else 0.0,
        "join_accuracy": float(edf["join_match"].mean()) if total else 0.0,
    }
    component_match_score = (
        sum(component_scores.values()) / len(component_scores) if component_scores else 0.0
    )

    lat = pd.to_numeric(edf["latency_ms"], errors="coerce").dropna()
    avg_latency_ms = float(lat.mean()) if not lat.empty else 0.0
    median_latency_ms = float(lat.median()) if not lat.empty else 0.0
    p95_latency_ms = float(lat.quantile(0.95)) if not lat.empty else 0.0

    summary = {
        "run_name": run_name,
        "examples": total,
        "correct": correct,
        "execution_accuracy": round(execution_accuracy, 4),
        "exact_match_accuracy": round(exact_match_accuracy, 4),
        "sql_validity_rate": round(sql_validity_rate, 4),
        "component_match_score": round(component_match_score, 4),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "median_latency_ms": round(median_latency_ms, 2),
        "p95_latency_ms": round(p95_latency_ms, 2),
        "result_mismatch_count": int((edf["outcome_type"] == "result_mismatch").sum()),
        "execution_error_count": pred_exec_error,
        "provider_error_count": int((edf["outcome_type"] == "provider_error").sum()),
        "parse_error_count": int((edf["outcome_type"] == "parse_error").sum()),
        "other_error_count": int((edf["outcome_type"] == "other").sum()),
        **{k: round(v, 4) for k, v in component_scores.items()},
    }

    per_db_df = (
        edf.groupby("db_id", dropna=False)
        .agg(
            examples=("db_id", "size"),
            correct=("exec_match", "sum"),
            execution_accuracy=("exec_match", "mean"),
            avg_latency_ms=("latency_ms", "mean"),
        )
        .reset_index()
        .sort_values(["execution_accuracy", "examples"], ascending=[True, False])
    )

    join_df = (
        edf.groupby("join_count", dropna=False)
        .agg(
            examples=("join_count", "size"),
            correct=("exec_match", "sum"),
            execution_accuracy=("exec_match", "mean"),
        )
        .reset_index()
        .sort_values("join_count")
    )

    complexity_df = (
        edf.groupby("complexity", dropna=False)
        .agg(
            examples=("complexity", "size"),
            correct=("exec_match", "sum"),
            execution_accuracy=("exec_match", "mean"),
        )
        .reset_index()
    )

    outcome_df = (
        edf.groupby("outcome_type", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    component_df = pd.DataFrame(
        {
            "component": ["SELECT", "WHERE", "GROUP BY", "ORDER BY", "JOIN"],
            "accuracy": [
                summary["select_accuracy"],
                summary["where_accuracy"],
                summary["group_accuracy"],
                summary["order_accuracy"],
                summary["join_accuracy"],
            ],
        }
    )

    return {
        "enriched_df": edf,
        "summary": summary,
        "summary_df": pd.DataFrame([summary]),
        "per_db_df": per_db_df,
        "join_df": join_df,
        "complexity_df": complexity_df,
        "outcome_df": outcome_df,
        "component_df": component_df,
    }


def summarize_eval_frames(named_frames: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for name, df in named_frames:
        payload = analyze_eval_frame(df, run_name=name)
        rows.append(payload["summary"])
    return pd.DataFrame(rows)