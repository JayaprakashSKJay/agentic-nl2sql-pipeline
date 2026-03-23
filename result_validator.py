from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd


def validate_result_frame(
    df: pd.DataFrame,
    nl_question: str,
    plan: Dict[str, Any] | None = None,
    max_rows_soft: int = 5000,
    max_cols_soft: int = 30,
) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    if df is None:
        return False, ['No dataframe returned']
    if df.empty:
        issues.append('Query returned 0 rows')
    if df.shape[0] > max_rows_soft:
        issues.append(f'Very large result set: {df.shape[0]} rows')
    if df.shape[1] > max_cols_soft:
        issues.append(f'Too many columns returned: {df.shape[1]} columns')
    if len(df.columns) != len(set(df.columns)):
        issues.append('Duplicate column names in result')
    if not df.empty:
        duplicate_ratio = float(df.duplicated().mean()) if len(df) > 1 else 0.0
        if duplicate_ratio > 0.95:
            issues.append('Result rows are almost entirely duplicates')
        null_ratio = float(df.isna().mean().mean())
        if null_ratio > 0.80:
            issues.append(f'High overall null ratio: {null_ratio:.2f}')
    q = (nl_question or '').lower()
    if any(tok in q for tok in ['count', 'total', 'sum', 'average', 'avg', 'minimum', 'maximum', 'max', 'min']):
        if not df.empty and df.shape[1] > 10:
            issues.append('Aggregation-style question returned too many columns')
    if plan:
        aggregations = plan.get('aggregations', []) or []
        if aggregations and not df.empty and df.shape[1] > 10:
            issues.append('Plan expected aggregation but result shape looks too wide')
    return len(issues) == 0, issues
