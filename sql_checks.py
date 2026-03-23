from __future__ import annotations

import re
from typing import List, Tuple

import sqlglot
from sqlglot import exp

from join_path_validator import validate_join_paths


def extract_table_names(sql_text: str) -> List[str]:
    try:
        ast = sqlglot.parse_one(sql_text)
    except Exception:
        return []
    names: List[str] = []
    for t in ast.find_all(exp.Table):
        if t.name:
            names.append(t.name)
    out, seen = [], set()
    for n in names:
        nn = n.strip('"`')
        if nn and nn not in seen:
            out.append(nn)
            seen.add(nn)
    return out


def check_allowlisted_tables(sql_text: str, allowed_tables: List[str]) -> Tuple[bool, str]:
    allowed = {t.lower() for t in (allowed_tables or [])}
    if not allowed:
        return True, 'OK (no allowlist)'
    used = extract_table_names(sql_text)
    bad = [t for t in used if t.lower() not in allowed]
    if bad:
        return False, f'Disallowed tables referenced: {bad}. Allowed: {sorted(allowed_tables)}'
    return True, 'OK'


def is_complex_question(question: str) -> bool:
    q = (question or '').lower()
    patterns = [
        'join', 'group by', 'having', 'subquery', 'nested', 'top', 'maximum',
        'minimum', 'average', 'count', 'distinct', 'between', 'after', 'before',
        'per ', 'each ', 'compare', 'trend', 'total', 'sum', 'avg'
    ]
    if any(p in q for p in patterns):
        return True
    return len(q.split()) >= 14


def has_order_by(sql_text: str) -> bool:
    return bool(re.search(r"\border\s+by\b", sql_text, re.IGNORECASE))


def run_sql_guardrails(sql_text: str, allowed_tables: List[str], full_schema_text: str, validate_joins: bool = True) -> Tuple[bool, str]:
    ok, msg = check_allowlisted_tables(sql_text, allowed_tables)
    if not ok:
        return False, msg
    tables = extract_table_names(sql_text)
    if validate_joins and len(tables) > 1 and (full_schema_text or '').strip():
        ok2, msg2 = validate_join_paths(sql_text, full_schema_text)
        if not ok2:
            return False, msg2
    return True, 'OK'
