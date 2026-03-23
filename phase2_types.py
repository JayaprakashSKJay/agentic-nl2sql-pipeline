from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SchemaChunk:
    id: str
    text: str
    source: str
    db_id: str = ''
    table_names: Optional[List[str]] = None


@dataclass
class SchemaPack:
    chunks: List[SchemaChunk]
    text: str
    allowed_tables: List[str]


@dataclass
class DecompositionPlan:
    subquestions: List[str]
    assumptions: List[str]
    join_hints: List[str]
    required_tables: Optional[List[str]] = None
    aggregations: Optional[List[str]] = None
    filters: Optional[List[str]] = None


@dataclass
class Phase2Config:
    kb_dir: str = './kb'
    namespace: str = 'default'
    top_k: int = 8
    max_iters: int = 4
    default_limit: int = 200
    max_db_retries: int = 3
    enable_learning: bool = True
    learning_mode: str = 'propose'
    enable_result_validation: bool = True
    enable_join_validation: bool = True
    hitl_require_approval: bool = False
