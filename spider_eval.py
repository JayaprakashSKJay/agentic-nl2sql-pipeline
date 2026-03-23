from __future__ import annotations

import json
import re
import time
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from nl2sql_runner import generate_sql_json, validate_sql
from spider_metrics import enrich_eval_frame, analyze_eval_frame


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
ENRICHED_DIR = Path("spider_testing/enriched")


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_spider_split(spider_root: str, split: str) -> Tuple[List[dict], Path]:
    """
    Supports common Spider layouts:
    - dev.json
    - train.json
    - train_spider.json
    - test.json
    """
    root = Path(spider_root).expanduser().resolve()
    split_l = split.lower()

    if split_l == "dev":
        candidates = [root / "dev.json"]
    elif split_l == "train":
        candidates = [root / "train.json", root / "train_spider.json"]
    elif split_l == "test":
        candidates = [root / "test.json"]
    else:
        raise ValueError("split must be 'dev', 'train', or 'test'")

    for path in candidates:
        if path.exists():
            return _read_json(path), path

    raise FileNotFoundError(f"Could not find split file for '{split}' under {root}")


def load_tables_json(spider_root: str) -> Tuple[List[dict], Path]:
    root = Path(spider_root).expanduser().resolve()

    candidates = [
        root / "tables.json",
        root / "test_tables.json",
    ]

    for path in candidates:
        if path.exists():
            return _read_json(path), path

    raise FileNotFoundError(f"Could not find tables.json under {root}")


def db_sqlite_path(spider_root: str, db_id: str) -> Path:
    root = Path(spider_root).expanduser().resolve()

    p1 = root / "database" / db_id / f"{db_id}.sqlite"
    p2 = root / "test_database" / db_id / f"{db_id}.sqlite"

    if p1.exists():
        return p1
    if p2.exists():
        return p2

    return p1


# -----------------------------------------------------------------------------
# Schema formatting (from tables.json only)
# -----------------------------------------------------------------------------
def schema_from_tables_json(tables_json: List[dict], db_id: str) -> str:
    """
    Create a compact schema string from tables.json only.
    """
    entry = next((x for x in tables_json if x.get("db_id") == db_id), None)
    if not entry:
        raise KeyError(f"db_id '{db_id}' not found in tables.json")

    table_names: List[str] = entry["table_names_original"]
    column_names: List[Tuple[int, str]] = entry["column_names_original"]
    column_types: List[str] = entry.get("column_types", [""] * len(column_names))
    primary_keys: List[int] = entry.get("primary_keys", [])
    foreign_keys: List[Tuple[int, int]] = entry.get("foreign_keys", [])

    pk_set = set(primary_keys)
    fk_map: Dict[int, int] = {}
    for a, b in foreign_keys:
        fk_map[a] = b

    def col_ref(col_idx: int) -> str:
        t_idx, c_name = column_names[col_idx]
        if t_idx == -1:
            return c_name
        return f"{table_names[t_idx]}.{c_name}"

    cols_by_table: Dict[int, List[int]] = {i: [] for i in range(len(table_names))}
    for i, (t_idx, _c_name) in enumerate(column_names):
        if t_idx == -1:
            continue
        cols_by_table[t_idx].append(i)

    lines: List[str] = ["Tables:"]
    for t_idx, t_name in enumerate(table_names):
        lines.append(f"{t_name}")
        for col_idx in cols_by_table[t_idx]:
            _, c_name = column_names[col_idx]
            c_type = column_types[col_idx] if col_idx < len(column_types) else ""
            tags: List[str] = []
            if col_idx in pk_set:
                tags.append("PK")
            if col_idx in fk_map:
                tags.append(f"FK → {col_ref(fk_map[col_idx])}")
            tag_txt = f" ({', '.join(tags)})" if tags else ""
            type_txt = f" [{c_type}]" if c_type else ""
            lines.append(f"  - {c_name}{type_txt}{tag_txt}")
        lines.append("")

    return "\n".join(lines).strip()


# -----------------------------------------------------------------------------
# SQLite execution + result compare
# -----------------------------------------------------------------------------
def _connect_sqlite(sqlite_path: Path) -> sqlite3.Connection:
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {sqlite_path}")
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    return conn


def _execute_sqlite(
    conn: sqlite3.Connection,
    sql: str,
    params: Optional[Dict[str, Any]] = None,
) -> List[Tuple]:
    cur = conn.cursor()
    if params:
        cur.execute(sql, params)
    else:
        cur.execute(sql)
    rows = cur.fetchall()
    return [tuple(r) for r in rows]


def _normalize_cell(x: Any) -> Any:
    if isinstance(x, float):
        return round(x, 6)
    return x


def _normalize_result(rows: List[Tuple]) -> List[str]:
    norm = [repr(tuple(_normalize_cell(x) for x in row)) for row in rows]
    norm.sort()
    return norm


_ORDER_BY_RE = re.compile(r"\border\s+by\b", re.IGNORECASE)


def execution_match(
    pred_sql: str,
    pred_params: Dict[str, Any],
    gold_sql: str,
    conn: sqlite3.Connection,
) -> Tuple[bool, str]:
    try:
        pred_rows = _execute_sqlite(conn, pred_sql, pred_params or None)
    except Exception as e:
        return False, f"Pred execution error: {e}"

    try:
        gold_rows = _execute_sqlite(conn, gold_sql, None)
    except Exception as e:
        return False, f"Gold execution error: {e}"

    pred_has_order = bool(_ORDER_BY_RE.search(pred_sql))
    gold_has_order = bool(_ORDER_BY_RE.search(gold_sql))
    order_sensitive = pred_has_order or gold_has_order

    if order_sensitive:
        pred_seq = [repr(tuple(_normalize_cell(x) for x in row)) for row in pred_rows]
        gold_seq = [repr(tuple(_normalize_cell(x) for x in row)) for row in gold_rows]
        return (pred_seq == gold_seq), ("OK" if pred_seq == gold_seq else "Result mismatch (order-sensitive)")
    else:
        pred_norm = _normalize_result(pred_rows)
        gold_norm = _normalize_result(gold_rows)
        return (pred_norm == gold_norm), ("OK" if pred_norm == gold_norm else "Result mismatch (set compare)")


# -----------------------------------------------------------------------------
# Persistence helpers
# -----------------------------------------------------------------------------
def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_enriched_spider_run(
    enriched_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    split: str,
    max_examples: int,
    provider_name: str,
) -> Tuple[Path, Path]:
    """
    Save enriched per-query CSV and per-run summary CSV.
    """
    ENRICHED_DIR.mkdir(parents=True, exist_ok=True)

    provider_slug = re.sub(r"[^a-zA-Z0-9]+", "_", provider_name.strip()).strip("_").lower()
    ts = _timestamp()

    enriched_path = ENRICHED_DIR / f"spider_eval_{split}_{max_examples}_{provider_slug}_{ts}_enriched.csv"
    summary_path = ENRICHED_DIR / f"spider_eval_{split}_{max_examples}_{provider_slug}_{ts}_summary.csv"

    enriched_df.to_csv(enriched_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    return enriched_path, summary_path


# -----------------------------------------------------------------------------
# Evaluation runner
# -----------------------------------------------------------------------------
@dataclass
class EvalConfig:
    spider_root: str
    split: str = "dev"
    provider_name: str = "Gemini 2.5 Flash"
    default_limit: int = 200
    max_examples: int = 100
    db_id_filter: Optional[str] = None
    auto_save_enriched: bool = True


def run_spider_eval(cfg: EvalConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Returns:
      - enriched per-example dataframe
      - summary dict
    """
    tables, tables_path = load_tables_json(cfg.spider_root)
    items, split_path = load_spider_split(cfg.spider_root, cfg.split)

    if cfg.db_id_filter:
        items = [x for x in items if x.get("db_id") == cfg.db_id_filter]

    items = items[: int(cfg.max_examples)]

    rows_out: List[Dict[str, Any]] = []
    correct = 0
    total = 0

    schema_cache: Dict[str, str] = {}
    conn_cache: Dict[str, sqlite3.Connection] = {}

    for i, it in enumerate(items, start=1):
        db_id = it["db_id"]
        question = it["question"]
        gold_sql = it["query"]

        if db_id not in schema_cache:
            schema_cache[db_id] = schema_from_tables_json(tables, db_id)

        if db_id not in conn_cache:
            conn_cache[db_id] = _connect_sqlite(db_sqlite_path(cfg.spider_root, db_id))

        schema_text = schema_cache[db_id]
        conn = conn_cache[db_id]

        t0 = time.time()
        pred_sql = ""
        pred_params: Dict[str, Any] = {}
        gen_error = ""
        exec_ok = False
        exec_msg = ""

        try:
            gen = generate_sql_json(
                provider_name=cfg.provider_name,
                dialect="sqlite",
                schema=schema_text,
                question=question,
                default_limit=int(cfg.default_limit),
            )
            pred_sql = gen.get("sql", "") or ""
            pred_params = gen.get("params", {}) or {}
            validate_sql(pred_sql)
        except Exception as e:
            gen_error = str(e)

        latency_ms = int((time.time() - t0) * 1000)

        if not gen_error and pred_sql:
            exec_ok, exec_msg = execution_match(pred_sql, pred_params, gold_sql, conn)

        total += 1
        if exec_ok:
            correct += 1

        rows_out.append(
            {
                "idx": i,
                "db_id": db_id,
                "question": question,
                "pred_sql": pred_sql,
                "pred_params": json.dumps(pred_params, ensure_ascii=False),
                "gold_sql": gold_sql,
                "exec_match": bool(exec_ok),
                "message": gen_error or exec_msg,
                "latency_ms": latency_ms,
            }
        )

    for c in conn_cache.values():
        try:
            c.close()
        except Exception:
            pass

    raw_df = pd.DataFrame(rows_out)

    # Enrich offline (no extra LLM calls)
    enriched_df = enrich_eval_frame(raw_df, dialect="sqlite")
    payload = analyze_eval_frame(enriched_df, run_name=f"spider_eval_{cfg.split}_{cfg.max_examples}", dialect="sqlite")
    summary = payload["summary"]

    summary.update(
        {
            "split": cfg.split,
            "provider": cfg.provider_name,
            "tables_file": str(tables_path),
            "split_file": str(split_path),
            "db_dir": str(Path(cfg.spider_root).expanduser().resolve() / "database"),
            "db_id_filter": cfg.db_id_filter or "",
            "default_limit": int(cfg.default_limit),
        }
    )

    # backward compatibility
    summary["accuracy"] = summary.get("execution_accuracy", 0.0)

    summary_df = pd.DataFrame([summary])

    if cfg.auto_save_enriched:
        enriched_path, summary_path = save_enriched_spider_run(
            enriched_df=enriched_df,
            summary_df=summary_df,
            split=cfg.split,
            max_examples=int(cfg.max_examples),
            provider_name=cfg.provider_name,
        )
        summary["enriched_csv_path"] = str(enriched_path)
        summary["summary_csv_path"] = str(summary_path)

    return enriched_df, summary