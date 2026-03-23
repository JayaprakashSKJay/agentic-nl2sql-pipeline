import pandas as pd
import re
import matplotlib.pyplot as plt

df = pd.read_csv("spider_eval_dev_100.csv")

def count_joins(sql):
    return len(re.findall(r"\bjoin\b", str(sql).lower()))

def has_nested(sql):
    return "select" in str(sql).lower()[1:]

def classify_complexity(sql):

    joins = count_joins(sql)

    if joins == 0:
        return "easy"
    elif joins == 1:
        return "medium"
    elif joins >= 2:
        return "hard"

    if has_nested(sql):
        return "extra_hard"

df["joins"] = df["gold_sql"].apply(count_joins)
df["complexity"] = df["gold_sql"].apply(classify_complexity)

accuracy_by_complexity = df.groupby("complexity")["exec_match"].mean()

print("\nAccuracy by Query Complexity")
print(accuracy_by_complexity)

accuracy_by_joins = df.groupby("joins")["exec_match"].mean()

print("\nAccuracy by Join Count")
print(accuracy_by_joins)

db_accuracy = df.groupby("db_id")["exec_match"].mean().sort_values()

print("\nAccuracy by Database")
print(db_accuracy)

plt.figure(figsize=(6,4))
accuracy_by_complexity.plot(kind="bar")
plt.title("Accuracy vs Query Complexity")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("complexity_accuracy.png")

plt.figure(figsize=(6,4))
accuracy_by_joins.plot(kind="bar")
plt.title("Accuracy vs Join Count")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("join_accuracy.png")

plt.figure(figsize=(8,4))
db_accuracy.plot(kind="bar")
plt.title("Accuracy by Database")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("db_accuracy.png")

print("\nCharts saved.")