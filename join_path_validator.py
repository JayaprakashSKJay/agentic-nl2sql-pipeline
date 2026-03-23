from __future__ import annotations

import re
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple

import sqlglot
from sqlglot import exp

FK_LINE_RE = re.compile(
    r"([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)\s*->\s*([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)",
    re.IGNORECASE,
)
DDL_FK_RE = re.compile(
    r"FOREIGN\s+KEY\s*\(\s*([A-Za-z_][\w]*)\s*\)\s*REFERENCES\s*([A-Za-z_][\w]*)\s*\(\s*([A-Za-z_][\w]*)\s*\)",
    re.IGNORECASE,
)

def parse_fk_edges_from_schema(schema_text: str) -> List[Tuple[str, str, str, str]]:
    edges: List[Tuple[str, str, str, str]] = []
    for m in FK_LINE_RE.finditer(schema_text or ""):
        lt, lc, rt, rc = m.groups()
        edges.append((lt, lc, rt, rc))

    current_table = None
    for line in (schema_text or '').splitlines():
        s = line.strip()
        if s.lower().startswith('table '):
            current_table = s.split()[1].strip('(`')
        ddl_match = DDL_FK_RE.search(s)
        if ddl_match and current_table:
            lc, rt, rc = ddl_match.groups()
            edges.append((current_table, lc, rt, rc))
    return edges

def build_table_graph(edges: List[Tuple[str, str, str, str]]) -> Dict[str, Set[str]]:
    graph: Dict[str, Set[str]] = defaultdict(set)
    for lt, _lc, rt, _rc in edges:
        graph[lt].add(rt)
        graph[rt].add(lt)
    return graph

def extract_tables_and_join_pairs(sql_text: str) -> Tuple[List[str], List[Tuple[str, str, str, str]]]:
    try:
        ast = sqlglot.parse_one(sql_text)
    except Exception:
        return [], []

    alias_to_table: Dict[str, str] = {}
    tables: List[str] = []
    for t in ast.find_all(exp.Table):
        table_name = t.name
        alias = t.alias_or_name
        alias_to_table[alias] = table_name
        if table_name not in tables:
            tables.append(table_name)

    join_pairs: List[Tuple[str, str, str, str]] = []
    for join in ast.find_all(exp.Join):
        on_expr = join.args.get('on')
        if not on_expr:
            continue
        eqs = list(on_expr.find_all(exp.EQ))
        for eq in eqs:
            left = eq.left
            right = eq.right
            if isinstance(left, exp.Column) and isinstance(right, exp.Column):
                lt_alias = left.table
                rt_alias = right.table
                lc = left.name
                rc = right.name
                lt = alias_to_table.get(lt_alias, lt_alias)
                rt = alias_to_table.get(rt_alias, rt_alias)
                if lt and rt and lc and rc:
                    join_pairs.append((lt, lc, rt, rc))
    return tables, join_pairs

def are_tables_connected(used_tables: List[str], fk_edges: List[Tuple[str, str, str, str]]) -> Tuple[bool, str]:
    if len(used_tables) <= 1:
        return True, 'OK'
    if not fk_edges:
        return True, 'No FK graph metadata available'
    graph = build_table_graph(fk_edges)
    start = used_tables[0]
    seen = {start}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nxt in graph.get(cur, set()):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    missing = [t for t in used_tables if t not in seen]
    if missing:
        return False, f'Tables not connected in FK graph: {missing}'
    return True, 'OK'

def validate_join_pairs(join_pairs: List[Tuple[str, str, str, str]], fk_edges: List[Tuple[str, str, str, str]]) -> Tuple[bool, str]:
    if not join_pairs or not fk_edges:
        return True, 'OK'
    normalized = set()
    for lt, lc, rt, rc in fk_edges:
        normalized.add((lt.lower(), lc.lower(), rt.lower(), rc.lower()))
        normalized.add((rt.lower(), rc.lower(), lt.lower(), lc.lower()))
    bad = []
    for lt, lc, rt, rc in join_pairs:
        key = (lt.lower(), lc.lower(), rt.lower(), rc.lower())
        if key not in normalized:
            bad.append((lt, lc, rt, rc))
    if bad:
        return False, f'JOIN conditions not aligned with FK graph: {bad}'
    return True, 'OK'

def validate_join_paths(sql_text: str, schema_text: str) -> Tuple[bool, str]:
    fk_edges = parse_fk_edges_from_schema(schema_text)
    used_tables, join_pairs = extract_tables_and_join_pairs(sql_text)
    ok_conn, msg_conn = are_tables_connected(used_tables, fk_edges)
    if not ok_conn:
        return False, msg_conn
    ok_join, msg_join = validate_join_pairs(join_pairs, fk_edges)
    if not ok_join:
        return False, msg_join
    return True, 'OK'
