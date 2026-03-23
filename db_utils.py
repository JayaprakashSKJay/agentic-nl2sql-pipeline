# db_utils.py
"""
Database utilities: execution + retry logic with logging.

Used by app.py / nl2sql_runner.py to:
- Run SQL queries against SQLAlchemy engine
- Retry transient DB/connection errors with exponential backoff
"""

import time
import random
from typing import Dict

import pandas as pd
import streamlit as st
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError


def run_query(engine, sql_query: str, params: Dict) -> pd.DataFrame:
    """
    Single execution of a SQL query using SQLAlchemy.
    """
    with engine.connect() as conn:
        result = conn.execute(text(sql_query), params)
        rows = result.fetchall()
        cols = result.keys()
        return pd.DataFrame(rows, columns=cols)


def is_transient_error(err: SQLAlchemyError) -> bool:
    """
    Heuristic to decide if an error is likely transient (worth retrying) or
    permanent (syntax error, missing table, permission issue, etc.).
    """
    msg = str(err.__cause__ or err).lower()

    transient_keywords = [
        "timeout",
        "timed out",
        "could not connect",
        "connection refused",
        "connection reset",
        "connection closed",
        "server closed the connection",
        "terminating connection",
        "deadlock detected",
        "serialization failure",
        "too many connections",
        "could not obtain lock",
    ]

    return any(k in msg for k in transient_keywords)


def run_query_with_retries(
    engine,
    sql_query: str,
    params: Dict,
    retries: int = 3,
    delay: float = 0.7,
) -> pd.DataFrame:
    """
    Executes SQL with retry logic and logs to Streamlit.

    - Retries ONLY on likely transient DB/connection errors.
    - Exponential backoff with jitter between attempts.
    - Logs attempt count and error messages in the UI.
    """
    last_err: SQLAlchemyError | None = None

    for attempt in range(1, retries + 1):
        try:
            if attempt > 1:
                st.info(f"🔁 DB retry {attempt}/{retries} for query execution…")

            return run_query(engine, sql_query, params)

        except SQLAlchemyError as err:
            last_err = err

            # If not transient → no point in retrying
            if not is_transient_error(err):
                st.error(
                    "❌ Query failed due to a non-retryable database error "
                    f"(attempt {attempt}/{retries})."
                )
                st.code(str(err.__cause__ or err))
                raise

            # If out of attempts → fail
            if attempt == retries:
                st.error(
                    f"❌ Query failed after {retries} attempts "
                    "(transient error persisted)."
                )
                st.code(str(err.__cause__ or err))
                raise

            # Exponential backoff + jitter
            sleep_time = delay * attempt + random.random() * 0.3
            st.warning(
                f"⚠️ Transient DB error on attempt {attempt}/{retries}:\n"
                f"{str(err.__cause__ or err)}\n"
                f"⏳ Retrying in {sleep_time:.2f} seconds…"
            )
            time.sleep(sleep_time)

    if last_err:
        raise last_err

    # Should not reach here
    raise RuntimeError("run_query_with_retries failed without raising last_err")
