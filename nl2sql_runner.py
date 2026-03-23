# nl2sql_runner.py
import json
import re
import itertools
from typing import Tuple, Dict, Any, List, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from llm_factory import get_llm
from prompts import build_system_prompt, build_few_shots, chat_prompt
from sql_checks import run_sql_guardrails
from result_validator import validate_result_frame

# -------------------------------------------------------------------
# Simple string literal replacer for Local T5 path (if used)
# -------------------------------------------------------------------
STRING_RE = re.compile(r"'([^']*)'")


def parameterize_sql_basic(sql_text: str) -> Tuple[str, dict]:
    """
    Very basic parameterization: replace quoted strings by :p1, :p2, ...
    Mainly used for Local T5 fallback.
    """
    params: Dict[str, Any] = {}
    counter = itertools.count(1)

    def repl(m):
        name = f"p{next(counter)}"
        params[name] = m.group(1)
        return f":{name}"

    sql_text = STRING_RE.sub(repl, sql_text)
    return sql_text, params


# -------------------------------------------------------------------
# Safety & validation
# -------------------------------------------------------------------
PROHIBITED = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


def validate_sql(sql_text: str) -> None:
    """
    Ensure SQL is:
    - Non-empty
    - Single statement (no dangerous ';')
    - Starts with SELECT or WITH (for CTE)
    - Does not contain dangerous DDL/DML keywords
    """
    if not sql_text or not sql_text.strip():
        raise ValueError("Empty SQL produced by model.")

    cleaned = sql_text.strip()
    # Remove wrapping backticks if any (MySQL weirdness)
    cleaned = cleaned.strip("`").strip()

    # Disallow multiple statements (allow a trailing semicolon, but no others)
    if ";" in cleaned[:-1]:
        raise ValueError("Multiple SQL statements are not allowed.")

    first_token_match = re.match(r"\s*([a-zA-Z]+)", cleaned)
    first_token = first_token_match.group(1).lower() if first_token_match else ""

    if first_token not in ("select", "with"):
        raise ValueError(
            f"Only SELECT/CTE statements allowed. Got starting keyword: '{first_token or 'None'}'"
        )

    if PROHIBITED.search(cleaned):
        raise ValueError("Prohibited keyword detected in SQL.")


def safe_parse_llm_json(raw: str) -> dict:
    """
    Parse STRICT JSON from LLM. Tolerates accidental ```json fences.
    """
    s = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL)
    if fence:
        s = fence.group(1)
    obj = json.loads(s)
    if not isinstance(obj, dict) or "sql" not in obj or "params" not in obj:
        raise ValueError("Model must return JSON with keys 'sql' and 'params'.")
    if not isinstance(obj["params"], dict):
        raise ValueError("'params' must be a JSON object/dict.")
    return obj


def add_default_limit(sql_text: str, limit: int) -> str:
    if re.search(r"\blimit\s+\d+\b", sql_text, re.IGNORECASE):
        return sql_text
    return f"{sql_text.rstrip().rstrip(';')} LIMIT {limit}"


# -------------------------------------------------------------------
# LLM invocation helper
# -------------------------------------------------------------------
def _invoke_llm(llm, prompt: str) -> str:
    """
    Wrapper to handle both ChatModels (AIMessage) and LLMs (str).
    """
    out = llm.invoke(prompt)
    content = getattr(out, "content", None)
    if content is not None:
        return content
    return str(out)


# -------------------------------------------------------------------
# SQL generation (one shot)
# -------------------------------------------------------------------
def generate_sql_json(
    provider_name: str,
    dialect: str,
    schema: str,
    question: str,
    default_limit: int,
) -> dict:
    """
    Generate one SQL candidate (JSON with 'sql' and 'params') for a given question.
    """
    # Local T5 path (optional, requires HF stack)
    if "local t5" in provider_name.lower() or "spider" in provider_name.lower():
        from llm_factory import _make_local_t5  # type: ignore[attr-defined]
        t5 = _make_local_t5()
        t5_prompt = (
            "Translate natural language to SQL.\n"
            f"Target dialect: {dialect}.\n"
            "Rules: ONE SELECT or WITH-only statement. No comments, no explanation.\n"
            "SCHEMA:\n"
            f"{schema}\n"
            "QUESTION:\n"
            f"{question}\n"
            "SQL:"
        )
        out = _invoke_llm(t5, t5_prompt)
        sql_raw = str(out).strip().rstrip(";")
        if not sql_raw.lower().startswith(("select", "with")):
            raise ValueError("Local model did not return a SELECT/CTE.")
        sql_raw = add_default_limit(sql_raw, default_limit)
        sql_param, params = parameterize_sql_basic(sql_raw)
        return {"sql": sql_param, "params": params}

    # Chat model path
    llm = get_llm(provider_name)
    system_prompt = build_system_prompt(dialect=dialect, default_limit=default_limit)
    few_shots = build_few_shots(dialect=dialect, default_limit=default_limit)

    prompt_text = chat_prompt.format(
        system_prompt=system_prompt,
        schema=schema,
        few_shots=few_shots,
        question=question,
    )

    raw = _invoke_llm(llm, prompt_text)
    parsed = safe_parse_llm_json(raw)
    parsed["sql"] = add_default_limit(parsed["sql"], default_limit)
    return parsed


def generate_sql_json_correction(
    provider_name: str,
    dialect: str,
    schema: str,
    original_question: str,
    previous_sql: str,
    error_msg: str,
    default_limit: int,
) -> dict:
    """
    Ask LLM to correct a failing SQL given the DB error.
    """
    llm = get_llm(provider_name)
    system_prompt = build_system_prompt(dialect=dialect, default_limit=default_limit)
    few_shots = build_few_shots(dialect=dialect, default_limit=default_limit)

    correction_question = (
        f"{original_question}\n\n"
        "The previous SQL you generated caused a database error.\n"
        "Previous SQL:\n"
        f"{previous_sql}\n\n"
        "Database error message:\n"
        f"{error_msg}\n\n"
        "Please return a corrected SQL query that obeys ALL rules."
    )

    prompt_text = chat_prompt.format(
        system_prompt=system_prompt,
        schema=schema,
        few_shots=few_shots,
        question=correction_question,
    )

    raw = _invoke_llm(llm, prompt_text)
    parsed = safe_parse_llm_json(raw)
    parsed["sql"] = add_default_limit(parsed["sql"], default_limit)
    return parsed


# -------------------------------------------------------------------
# DB execution with retries
# -------------------------------------------------------------------
def execute_sql_with_retries(
    engine,
    sql_text: str,
    params: dict,
    max_db_retries: int = 3,
) -> Tuple[pd.DataFrame, int]:
    """
    Execute SQL with up to max_db_retries retries on SQLAlchemyError.
    Returns (df, retries_used).
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, max_db_retries + 1):
        try:
            with engine.connect() as conn:
                result = conn.execute(text(sql_text), params)
                rows = result.fetchall()
                cols = result.keys()
                df = pd.DataFrame(rows, columns=cols)
                return df, attempt
        except SQLAlchemyError as err:
            last_err = err
            # On last attempt, re-raise
            if attempt == max_db_retries:
                raise
    # Safety net – should not reach here
    if last_err:
        raise last_err
    return pd.DataFrame(), max_db_retries


# -------------------------------------------------------------------
# Orchestrator: semantic retries + DB retries
# -------------------------------------------------------------------
def run_nl2sql_flow(
    engine,
    provider_name: str,
    db_type: str,
    schema_text: str,
    nl_question: str,
    default_limit: int = 200,
    max_semantic_retries: int = 3,  # when no rows returned
    max_db_retries: int = 3,        # for transient DB errors
) -> Dict[str, Any]:
    """
    Full pipeline:
      1. Generate SQL from NL question
      2. Validate SQL
      3. Execute with DB retries
      4. If no rows, regenerate (semantic retry) up to max_semantic_retries
      5. If DB error, one correction attempt via LLM
    Returns dict with:
      - sql
      - params
      - df
      - logs (list of dict)
      - semantic_attempts
      - db_retries_total
    """
    dbt = (db_type or '').lower()
    if dbt.startswith('sqlite'):
        dialect = 'sqlite'
    elif dbt.startswith('post'):
        dialect = 'postgresql'
    else:
        dialect = 'mysql'
    logs: List[Dict[str, Any]] = []

    final_sql = ""
    final_params: Dict[str, Any] = {}
    final_df = pd.DataFrame()
    total_db_retries_used = 0

    question_base = nl_question
    correction_used = False

    for semantic_attempt in range(1, max_semantic_retries + 1):
        question_for_llm = question_base
        if semantic_attempt > 1:
            # Augment question with hint that previous attempts returned no rows
            question_for_llm = (
                f"{nl_question}\n\n"
                "Note: The previous SQL query returned 0 rows. "
                "Please slightly broaden or relax filters while preserving intent."
            )

        # 1) Generate SQL
        try:
            gen = generate_sql_json(
                provider_name=provider_name,
                dialect=dialect,
                schema=schema_text,
                question=question_for_llm,
                default_limit=default_limit,
            )
            sql_text = gen["sql"]
            params = gen.get("params", {}) or {}

            validate_sql(sql_text)

            logs.append(
                {
                    "stage": "generation",
                    "semantic_attempt": semantic_attempt,
                    "sql_preview": sql_text[:200],
                    "params": params,
                }
            )

        except Exception as e:
            logs.append(
                {
                    "stage": "generation_error",
                    "semantic_attempt": semantic_attempt,
                    "error": str(e),
                }
            )
            raise RuntimeError(f"Failed to produce safe SQL on attempt {semantic_attempt}: {e}")

        # 2) Execute with DB retries
        try:
            df, db_tries = execute_sql_with_retries(
                engine=engine,
                sql_text=sql_text,
                params=params,
                max_db_retries=max_db_retries,
            )
            total_db_retries_used += db_tries

            logs.append(
                {
                    "stage": "execution",
                    "semantic_attempt": semantic_attempt,
                    "db_retries_used": db_tries,
                    "rows_returned": int(df.shape[0]),
                }
            )

            # Success
            final_sql = sql_text
            final_params = params
            final_df = df

            if not df.empty:
                break  # we are done

            # df is empty
            if semantic_attempt == max_semantic_retries:
                # No more semantic retries; return as-is
                break
            # else: loop again, with "no rows" context

        except SQLAlchemyError as err:
            err_msg = str(err.__cause__ or err)
            logs.append(
                {
                    "stage": "db_error",
                    "semantic_attempt": semantic_attempt,
                    "db_retries_used": max_db_retries,
                    "error": err_msg,
                }
            )

            # If Local T5, don't try semantic correction; just raise
            if "local t5" in provider_name.lower() or "spider" in provider_name.lower():
                raise RuntimeError(f"Query failed after DB retries: {err_msg}")

            # One correction attempt via LLM for DB error
            if not correction_used:
                correction_used = True
                try:
                    corr = generate_sql_json_correction(
                        provider_name=provider_name,
                        dialect=dialect,
                        schema=schema_text,
                        original_question=nl_question,
                        previous_sql=sql_text,
                        error_msg=err_msg,
                        default_limit=default_limit,
                    )
                    sql_corr = corr["sql"]
                    params_corr = corr.get("params", {}) or {}

                    validate_sql(sql_corr)

                    logs.append(
                        {
                            "stage": "correction_generation",
                            "semantic_attempt": semantic_attempt,
                            "sql_preview": sql_corr[:200],
                            "params": params_corr,
                        }
                    )

                    df_corr, db_tries_corr = execute_sql_with_retries(
                        engine=engine,
                        sql_text=sql_corr,
                        params=params_corr,
                        max_db_retries=max_db_retries,
                    )
                    total_db_retries_used += db_tries_corr

                    logs.append(
                        {
                            "stage": "correction_execution",
                            "semantic_attempt": semantic_attempt,
                            "db_retries_used": db_tries_corr,
                            "rows_returned": int(df_corr.shape[0]),
                        }
                    )

                    final_sql = sql_corr
                    final_params = params_corr
                    final_df = df_corr
                    break  # accept corrected query result

                except Exception as e2:
                    logs.append(
                        {
                            "stage": "correction_error",
                            "semantic_attempt": semantic_attempt,
                            "error": str(e2),
                        }
                    )
                    raise RuntimeError(
                        f"Correction attempt failed after DB error. Original error: {err_msg}; correction error: {e2}"
                    )
            else:
                # Already tried correction once; give up
                raise RuntimeError(f"Query failed after DB retries and correction: {err_msg}")

    return {
        "sql": final_sql,
        "params": final_params,
        "df": final_df,
        "logs": logs,
        "semantic_attempts": semantic_attempt,
        "db_retries_total": total_db_retries_used,
        "dialect": dialect,
    }


# -------------------------------------------------------------------
# Phase 2: RAG + decomposition + stronger refinement + learning
# -------------------------------------------------------------------

def run_nl2sql_flow_phase2(
    engine,
    provider_name: str,
    db_type: str,
    full_schema_text: str,
    nl_question: str,
    cfg=None,
):
    from phase2_types import Phase2Config
    from tracing_utils import new_trace_session
    from schema_kb import build_schema_kb, retrieve_schema_chunks, retrieve_learned_fewshots, append_learned_example
    from schema_pack_utils import build_schema_pack
    from sql_checks import is_complex_question
    from phase2_agents import decomposer_agent, sql_generator_agent, refiner_agent

    if cfg is None:
        cfg = Phase2Config()
    if not isinstance(cfg, Phase2Config):
        cfg = Phase2Config(**dict(cfg))

    dbt = (db_type or '').lower()
    if dbt.startswith('sqlite'):
        dialect = 'sqlite'
    elif dbt.startswith('post'):
        dialect = 'postgresql'
    else:
        dialect = 'mysql'

    trace = new_trace_session()
    logs: List[Dict[str, Any]] = []

    kb_info = build_schema_kb(
        schema_text=full_schema_text,
        kb_dir=cfg.kb_dir,
        namespace=cfg.namespace,
        source='schema:live',
    )
    trace.log('kb_built', kb_info)

    chunks = retrieve_schema_chunks(
        question=nl_question,
        kb_dir=cfg.kb_dir,
        namespace=cfg.namespace,
        top_k=int(cfg.top_k),
    )
    schema_pack = build_schema_pack(chunks)
    trace.log('schema_retrieved', {
        'top_k': cfg.top_k,
        'chunks': len(chunks),
        'allowed_tables': schema_pack.allowed_tables,
    })

    plan = None
    if is_complex_question(nl_question):
        try:
            plan = decomposer_agent(
                provider_name=provider_name,
                question=nl_question,
                schema_pack=schema_pack.text,
                dialect=dialect,
            )
            trace.log('decomposition_plan', plan)
        except Exception as e:
            trace.log('decomposition_error', {'error': str(e)})
            plan = None

    learned = []
    if cfg.enable_learning:
        learned = retrieve_learned_fewshots(
            question=nl_question,
            kb_dir=cfg.kb_dir,
            namespace=cfg.namespace,
            top_k=3,
        )
        if learned:
            trace.log('learned_retrieved', {'count': len(learned)})

    final_sql = ''
    final_params: Dict[str, Any] = {}
    final_df = pd.DataFrame()
    total_db_retries_used = 0
    last_feedback = ''
    learning_proposal = None
    result_validation = {'ok': True, 'issues': []}

    for it in range(1, int(cfg.max_iters) + 1):
        trace.log('iter_start', {'iter': it})
        try:
            if it == 1:
                gen = sql_generator_agent(
                    provider_name=provider_name,
                    dialect=dialect,
                    schema_pack=schema_pack.text,
                    question=nl_question,
                    default_limit=int(cfg.default_limit),
                    extra_fewshots=learned,
                    plan_json=plan,
                )
            else:
                gen = refiner_agent(
                    provider_name=provider_name,
                    dialect=dialect,
                    schema_pack=schema_pack.text,
                    question=nl_question,
                    previous_sql=final_sql,
                    error_or_feedback=last_feedback,
                    default_limit=int(cfg.default_limit),
                )

            sql_text = gen['sql']
            params = gen.get('params', {}) or {}

            logs.append({'stage': 'generation', 'iter': it, 'sql_preview': sql_text[:200], 'params': params})
            trace.log('sql_generated', {'iter': it, 'sql': sql_text, 'params': params})

            ok, msg = run_sql_guardrails(
                sql_text,
                schema_pack.allowed_tables,
                full_schema_text if cfg.enable_join_validation else '',
                validate_joins=bool(cfg.enable_join_validation),
            )
            if not ok:
                final_sql, final_params = sql_text, params
                last_feedback = f'Guardrail failed: {msg}'
                logs.append({'stage': 'guardrail_failed', 'iter': it, 'message': msg})
                trace.log('guardrail_failed', {'iter': it, 'message': msg})
                continue

            if cfg.hitl_require_approval and it == 1:
                trace.log('awaiting_approval', {'sql': sql_text, 'params': params})
                return {
                    'sql': sql_text,
                    'params': params,
                    'df': pd.DataFrame(),
                    'logs': logs,
                    'db_retries_total': total_db_retries_used,
                    'dialect': dialect,
                    'phase': 'phase2',
                    'status': 'awaiting_approval',
                    'schema_pack': schema_pack.text,
                    'allowed_tables': schema_pack.allowed_tables,
                    'plan': plan or {},
                    'trace_path': str(trace.path),
                    'trace_dir': str(trace.out_dir),
                }

            df, db_tries = execute_sql_with_retries(
                engine=engine,
                sql_text=sql_text,
                params=params,
                max_db_retries=int(cfg.max_db_retries),
            )
            total_db_retries_used += db_tries
            logs.append({'stage': 'execution', 'iter': it, 'db_retries_used': db_tries, 'rows_returned': int(df.shape[0])})
            trace.log('sql_executed', {'iter': it, 'rows': int(df.shape[0])})

            final_sql, final_params, final_df = sql_text, params, df

            if cfg.enable_result_validation:
                ok_res, issues = validate_result_frame(df, nl_question, plan=plan)
                result_validation = {'ok': ok_res, 'issues': issues}
                trace.log('result_validation', result_validation)
                if (not ok_res) and it < int(cfg.max_iters):
                    last_feedback = 'Result validation issues: ' + '; '.join(issues)
                    continue

            if df.empty and it < int(cfg.max_iters):
                last_feedback = (
                    'Query executed successfully but returned 0 rows. '
                    'Relax overly strict filters while preserving intent.'
                )
                trace.log('empty_result', {'iter': it})
                continue

            break

        except SQLAlchemyError as err:
            err_msg = str(err.__cause__ or err)
            last_feedback = f'DB execution error: {err_msg}'
            logs.append({'stage': 'db_error', 'iter': it, 'error': err_msg})
            trace.log('db_error', {'iter': it, 'error': err_msg})
            continue

        except Exception as e:
            last_feedback = str(e)
            logs.append({'stage': 'phase2_error', 'iter': it, 'error': str(e)})
            trace.log('phase2_error', {'iter': it, 'error': str(e)})
            break

    learned_saved = False
    if cfg.enable_learning and final_sql:
        if cfg.learning_mode.lower() == 'always':
            append_learned_example(
                nl_question=nl_question,
                sql_text=final_sql,
                kb_dir=cfg.kb_dir,
                namespace=cfg.namespace,
            )
            learned_saved = True
        elif cfg.learning_mode.lower() == 'propose':
            learning_proposal = {
                'nl_question': nl_question,
                'sql_text': final_sql,
                'kb_dir': cfg.kb_dir,
                'namespace': cfg.namespace,
            }

    trace.log('run_end', {
        'learned_saved': learned_saved,
        'result_validation': result_validation,
    })

    return {
        'sql': final_sql,
        'params': final_params,
        'df': final_df,
        'logs': logs,
        'db_retries_total': total_db_retries_used,
        'dialect': dialect,
        'phase': 'phase2',
        'schema_pack': schema_pack.text,
        'allowed_tables': schema_pack.allowed_tables,
        'plan': plan or {},
        'trace_path': str(trace.path),
        'trace_dir': str(trace.out_dir),
        'learning_proposal': learning_proposal,
        'result_validation': result_validation,
    }

