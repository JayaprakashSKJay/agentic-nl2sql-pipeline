from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from llm_factory import get_llm
from nl2sql_runner import safe_parse_llm_json, validate_sql, add_default_limit
from prompts import build_system_prompt, build_few_shots, chat_prompt


def _invoke(llm, prompt: str) -> str:
    out = llm.invoke(prompt)
    content = getattr(out, 'content', None)
    return content if content is not None else str(out)


def _parse_strict_json(raw: str) -> Dict[str, Any]:
    s = raw.strip()
    fence = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', s, re.DOTALL)
    if fence:
        s = fence.group(1)
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError('Model must return a JSON object')
    return obj


def decomposer_agent(provider_name: str, question: str, schema_pack: str, dialect: str) -> Dict[str, Any]:
    llm = get_llm(provider_name)
    prompt = f"""You are a planning agent for NL-to-SQL.

Given the user's question and a relevant schema context pack, produce a decomposition plan.

Output STRICT JSON only with keys:
- subquestions: list[str]
- assumptions: list[str]
- join_hints: list[str]
- required_tables: list[str]
- filters: list[str]
- aggregations: list[str]

Dialect: {dialect}

SCHEMA CONTEXT PACK:
{schema_pack}

QUESTION:
{question}
"""
    raw = _invoke(llm, prompt)
    obj = _parse_strict_json(raw)
    for k in ('subquestions', 'assumptions', 'join_hints', 'required_tables', 'filters', 'aggregations'):
        if not isinstance(obj.get(k), list):
            obj[k] = []
    return obj


def sql_generator_agent(
    provider_name: str,
    dialect: str,
    schema_pack: str,
    question: str,
    default_limit: int,
    extra_fewshots: List[str] | None = None,
    plan_json: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    llm = get_llm(provider_name)
    system_prompt = build_system_prompt(dialect=dialect, default_limit=default_limit)
    few_shots = build_few_shots(dialect=dialect, default_limit=default_limit)
    plan_block = ''
    if plan_json:
        plan_block = '\n\nPLAN (from decomposer):\n' + json.dumps(plan_json, ensure_ascii=False)
    learned_block = ''
    if extra_fewshots:
        learned_block = '\n\nLEARNED EXAMPLES (use as guidance, do not copy blindly):\n' + '\n\n'.join(extra_fewshots)
    q = f'{question}{plan_block}{learned_block}'
    prompt_text = chat_prompt.format(
        system_prompt=system_prompt,
        schema=schema_pack,
        few_shots=few_shots,
        question=q,
    )
    raw = _invoke(llm, prompt_text)
    parsed = safe_parse_llm_json(raw)
    parsed['sql'] = add_default_limit(parsed['sql'], default_limit)
    validate_sql(parsed['sql'])
    return parsed


def refiner_agent(
    provider_name: str,
    dialect: str,
    schema_pack: str,
    question: str,
    previous_sql: str,
    error_or_feedback: str,
    default_limit: int,
) -> Dict[str, Any]:
    llm = get_llm(provider_name)
    system_prompt = build_system_prompt(dialect=dialect, default_limit=default_limit)
    few_shots = build_few_shots(dialect=dialect, default_limit=default_limit)
    refine_question = (
        f'{question}\n\n'
        'We are in an iterative refinement loop.\n'
        'Previous SQL:\n'
        f'{previous_sql}\n\n'
        'Feedback (DB error and/or semantic check failures):\n'
        f'{error_or_feedback}\n\n'
        'Return corrected STRICT JSON with keys sql and params.\n'
        'IMPORTANT: Use ONLY tables/columns from the schema pack.'
    )
    prompt_text = chat_prompt.format(
        system_prompt=system_prompt,
        schema=schema_pack,
        few_shots=few_shots,
        question=refine_question,
    )
    raw = _invoke(llm, prompt_text)
    parsed = safe_parse_llm_json(raw)
    parsed['sql'] = add_default_limit(parsed['sql'], default_limit)
    validate_sql(parsed['sql'])
    return parsed
