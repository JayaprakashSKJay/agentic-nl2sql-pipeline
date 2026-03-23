# prompts.py
import json
from langchain_core.prompts import PromptTemplate

# -------------------------------------------------------------------
# Base SYSTEM prompt (dialect-aware, limit-aware)
# -------------------------------------------------------------------
SYSTEM_PROMPT_TEMPLATE = """
You are an expert SQL generator.

You MUST ALWAYS output STRICT JSON:
{{"sql": "...", "params": {{ ... }} }}

SQL RULES:
- ALWAYS return a SINGLE SELECT statement (or CTE starting with WITH).
- NEVER return explanations, comments, natural-language sentences, or error messages.
- NEVER return "No data", "None", "Error", or similar text in "sql".
- Use ONLY the tables and columns from the provided schema.
- Use named parameters (:p1, :p2, ...) and put literal values ONLY in "params".
- If the user doesn't specify LIMIT, append LIMIT {default_limit}.
- Target database DIALECT: {dialect}.
  - For PostgreSQL: use DATE_PART / EXTRACT, standard Postgres syntax.
  - For MySQL: use YEAR(), MONTH(), standard MySQL syntax.
  - For SQLite: use strftime('%Y', date_col), julianday(), and SQLite-compatible functions.
- If unsure, make your best guess BUT ALWAYS return a valid SELECT query.

Output STRICT JSON only, no code fences:
{{"sql": "SELECT ...", "params": {{"p1": "value"}}}}
"""

# -------------------------------------------------------------------
# Few-shot examples – rendered per dialect
# -------------------------------------------------------------------
def build_few_shots(dialect: str, default_limit: int) -> str:
    """
    Return a string of few-shot examples adapted for the given dialect.
    """
    dialect = (dialect or "").lower()

    if dialect == "mysql":
        examples = [
            {
                "schema": """Table customers(id INT, company VARCHAR(255), city VARCHAR(255), email VARCHAR(255))
Table orders(id INT, customer_id INT, dt DATE, total DECIMAL(10,2))
Relationship: orders.customer_id -> customers.id""",
                "q": "Total revenue by company with email and city for 2023.",
                "a": {
                    "sql": (
                        "SELECT c.company, c.city, c.email, SUM(o.total) AS revenue "
                        "FROM customers c "
                        "JOIN orders o ON o.customer_id = c.id "
                        "WHERE YEAR(o.dt) = :p1 "
                        "GROUP BY c.company, c.city, c.email "
                        "ORDER BY revenue DESC "
                        f"LIMIT {default_limit}"
                    ),
                    "params": {"p1": 2023},
                },
            },
            {
                "schema": """Table customers(id INT, company VARCHAR(255), city VARCHAR(255), email VARCHAR(255))
Table orders(id INT, customer_id INT, dt DATE, total DECIMAL(10,2))""",
                "q": "Show ABC Corp revenue in April 2024.",
                "a": {
                    "sql": (
                        "SELECT SUM(o.total) AS revenue "
                        "FROM customers c "
                        "JOIN orders o ON o.customer_id = c.id "
                        "WHERE c.company = :p1 "
                        "AND o.dt >= :p2 AND o.dt < :p3 "
                        f"LIMIT {default_limit}"
                    ),
                    "params": {"p1": "ABC Corp", "p2": "2024-04-01", "p3": "2024-05-01"},
                },
            },
        ]

    elif dialect == "sqlite":
        examples = [
            {
                "schema": """Table customers(id INTEGER, company TEXT, city TEXT, email TEXT)
Table orders(id INTEGER, customer_id INTEGER, dt TEXT, total REAL)
Relationship: orders.customer_id -> customers.id""",
                "q": "Total revenue by company for 2023.",
                "a": {
                    "sql": (
                        "SELECT c.company, SUM(o.total) AS revenue "
                        "FROM customers c "
                        "JOIN orders o ON o.customer_id = c.id "
                        "WHERE strftime('%Y', o.dt) = :p1 "
                        "GROUP BY c.company "
                        "ORDER BY revenue DESC "
                        f"LIMIT {default_limit}"
                    ),
                    "params": {"p1": "2023"},
                },
            },
            {
                "schema": """Table concert(concert_id INTEGER, concert_name TEXT, stadium_id INTEGER)
Table stadium(stadium_id INTEGER, name TEXT, capacity INTEGER)
Relationship: concert.stadium_id -> stadium.stadium_id""",
                "q": "List concert names and stadium names.",
                "a": {
                    "sql": (
                        "SELECT c.concert_name, s.name AS stadium_name "
                        "FROM concert c "
                        "JOIN stadium s ON s.stadium_id = c.stadium_id "
                        f"LIMIT {default_limit}"
                    ),
                    "params": {},
                },
            },
        ]

    else:
        examples = [
            {
                "schema": """Table customers(id INT, company TEXT, city TEXT, email TEXT)
Table orders(id INT, customer_id INT, dt DATE, total NUMERIC)
Relationship: orders.customer_id -> customers.id""",
                "q": "Total revenue by company with email and city for 2023.",
                "a": {
                    "sql": (
                        "SELECT c.company, c.city, c.email, SUM(o.total) AS revenue "
                        "FROM customers c "
                        "JOIN orders o ON o.customer_id = c.id "
                        "WHERE DATE_PART('year', o.dt) = :p1 "
                        "GROUP BY c.company, c.city, c.email "
                        "ORDER BY revenue DESC "
                        f"LIMIT {default_limit}"
                    ),
                    "params": {"p1": 2023},
                },
            },
            {
                "schema": """Table customers(id INT, company TEXT, city TEXT, email TEXT)
Table orders(id INT, customer_id INT, dt DATE, total NUMERIC)""",
                "q": "Show ABC Corp revenue in April 2024.",
                "a": {
                    "sql": (
                        "SELECT SUM(o.total) AS revenue "
                        "FROM customers c "
                        "JOIN orders o ON o.customer_id = c.id "
                        "WHERE c.company = :p1 "
                        "AND o.dt >= :p2 AND o.dt < :p3 "
                        f"LIMIT {default_limit}"
                    ),
                    "params": {"p1": "ABC Corp", "p2": "2024-04-01", "p3": "2024-05-01"},
                },
            },
        ]

    parts = []
    for ex in examples:
        parts.append(
            f"""Schema:
{ex['schema']}
Q: {ex['q']}
A: {json.dumps(ex['a'], ensure_ascii=False)}"""
        )

    # Separate examples with a blank line
    return "\n\n".join(parts)


# -------------------------------------------------------------------
# Chat prompt template
# -------------------------------------------------------------------
CHAT_TEMPLATE = """{system_prompt}

SCHEMA:
{schema}

EXAMPLES:
{few_shots}

USER QUESTION:
{question}

If additional context is provided about previous attempts or no results, respect it but
STILL output STRICT JSON with keys "sql" and "params".
"""

chat_prompt = PromptTemplate(
    input_variables=["system_prompt", "schema", "few_shots", "question"],
    template=CHAT_TEMPLATE,
)


def build_system_prompt(dialect: str, default_limit: int) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(dialect=dialect, default_limit=default_limit)
