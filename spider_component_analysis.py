import pandas as pd
import re

df = pd.read_csv("spider_eval_dev_100.csv")

def extract_select(sql):
    m = re.search(r"select (.*?) from", sql.lower())
    return m.group(1) if m else ""

def extract_where(sql):
    m = re.search(r"where (.*?)(group by|order by|limit|$)", sql.lower())
    return m.group(1) if m else ""

def extract_group(sql):
    m = re.search(r"group by (.*?)(order by|limit|$)", sql.lower())
    return m.group(1) if m else ""

def extract_order(sql):
    m = re.search(r"order by (.*?)(limit|$)", sql.lower())
    return m.group(1) if m else ""

def count_joins(sql):
    return len(re.findall(r"\bjoin\b", sql.lower()))

def component_match(a, b):
    return str(a).strip() == str(b).strip()

df["select_match"] = df.apply(
    lambda r: component_match(
        extract_select(str(r["pred_sql"])),
        extract_select(str(r["gold_sql"]))
    ), axis=1
)

df["where_match"] = df.apply(
    lambda r: component_match(
        extract_where(str(r["pred_sql"])),
        extract_where(str(r["gold_sql"]))
    ), axis=1
)

df["group_match"] = df.apply(
    lambda r: component_match(
        extract_group(str(r["pred_sql"])),
        extract_group(str(r["gold_sql"]))
    ), axis=1
)

df["order_match"] = df.apply(
    lambda r: component_match(
        extract_order(str(r["pred_sql"])),
        extract_order(str(r["gold_sql"]))
    ), axis=1
)

df["join_match"] = df.apply(
    lambda r: count_joins(str(r["pred_sql"])) ==
              count_joins(str(r["gold_sql"])),
    axis=1
)

summary = {
    "SELECT_accuracy": df["select_match"].mean(),
    "WHERE_accuracy": df["where_match"].mean(),
    "GROUP_BY_accuracy": df["group_match"].mean(),
    "ORDER_BY_accuracy": df["order_match"].mean(),
    "JOIN_accuracy": df["join_match"].mean(),
}

summary["component_match_score"] = sum(summary.values()) / len(summary)

print("\nComponent Matching Metrics\n")

for k, v in summary.items():
    print(f"{k}: {v:.3f}")