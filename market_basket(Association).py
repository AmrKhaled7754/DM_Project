"""
Task 3 - ASSOCIATION RULES: Product Recommendation System (UCI Online Retail)
Apriori -> rules (support, confidence, lift) -> recommend(product) function.
"""
# %% Imports & config
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from mlxtend.frequent_patterns import apriori, association_rules

from data_loader import load_retail

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42
COUNTRY = "United Kingdom"   # keeps the data small enough for a normal laptop
TOP_ITEMS = 150              # keep only the most frequent products (memory/speed)
MIN_SUPPORT = 0.02           # lower it (e.g. 0.01) for more rules, but it is slower
MIN_LIFT = 1.0

# %% A. Dataset
df = load_retail()
print(df.shape)
print(df.head())
df.info()

# %% B. Preprocessing
print("\nMissing per column:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

df = df.drop_duplicates()
df = df.dropna(subset=["InvoiceNo", "Description"])
df["InvoiceNo"] = df["InvoiceNo"].astype(str)
df = df[~df["InvoiceNo"].str.startswith("C")]          # cancelled invoices
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]   # returns / bad rows
df["Description"] = df["Description"].str.strip()
df = df[df["Country"] == COUNTRY]
print("After cleaning:", df.shape)

# keep the most frequent products only
item_counts = df.groupby("Description")["InvoiceNo"].nunique().sort_values(ascending=False)
top_items = item_counts.head(TOP_ITEMS).index
df = df[df["Description"].isin(top_items)]

# basket format: rows = invoices, columns = products, values = True/False
basket = (df.groupby(["InvoiceNo", "Description"])["Quantity"].sum().unstack(fill_value=0) > 0)
basket = basket[basket.sum(axis=1) >= 2]    # an invoice with 1 item gives no rule
print("Basket shape (invoices x products):", basket.shape)

# %% G. Exploration visuals
plt.figure(figsize=(9, 5))
item_counts.head(10).sort_values().plot(kind="barh")
plt.title("Top 10 products (number of invoices)")
plt.savefig(OUT / "retail_top_products.png", dpi=150, bbox_inches="tight")
plt.show()

plt.figure(figsize=(6, 4))
sns.histplot(basket.sum(axis=1), bins=30)
plt.title("Items per invoice")
plt.xlabel("Number of distinct products")
plt.savefig(OUT / "retail_items_per_invoice.png", dpi=150, bbox_inches="tight")
plt.show()

# %% C. Split invoices 80/20: mine rules on train, check them on test
train_b, test_b = train_test_split(basket, test_size=0.2, random_state=RANDOM_STATE)

# %% Apriori + rules
freq = apriori(train_b, min_support=MIN_SUPPORT, use_colnames=True)
print("Frequent itemsets:", len(freq))
try:   # newer mlxtend versions want num_itemsets
    rules = association_rules(freq, num_itemsets=len(train_b), metric="lift", min_threshold=MIN_LIFT)
except TypeError:
    rules = association_rules(freq, metric="lift", min_threshold=MIN_LIFT)
rules = rules.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)
print("Rules:", len(rules))
if rules.empty:
    raise SystemExit("No rules found. Lower MIN_SUPPORT or raise TOP_ITEMS.")


def label(fs):
    return ", ".join(sorted(fs))


rules_out = rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
rules_out["antecedents"] = rules_out["antecedents"].apply(label)
rules_out["consequents"] = rules_out["consequents"].apply(label)
rules_out.to_csv(OUT / "retail_rules.csv", index=False)
print(rules_out.head(10).round(3).to_string())

# %% F. Metrics summary
print(rules[["support", "confidence", "lift"]].describe().round(3))

# %% G. Visualizations of the rules
plt.figure(figsize=(7, 5))
sc = plt.scatter(rules["support"], rules["confidence"], c=rules["lift"], cmap="viridis", alpha=0.7)
plt.colorbar(sc, label="Lift")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Association rules: support vs confidence (color = lift)")
plt.savefig(OUT / "retail_rules_scatter.png", dpi=150, bbox_inches="tight")
plt.show()

top = rules.head(10).copy()
top["rule"] = top["antecedents"].apply(label) + "  ->  " + top["consequents"].apply(label)
top.sort_values("lift").plot(x="rule", y="lift", kind="barh", legend=False, figsize=(11, 5),
                             title="Top 10 rules by lift")
plt.savefig(OUT / "retail_top_rules.png", dpi=150, bbox_inches="tight")
plt.show()

# %% D. Recommendation function (this is the "predict" step for association rules)
def recommend(items, rules_df=rules, n=5):
    """Given a list/set of products in the basket, return up to n recommended products."""
    items = set(items)
    cand = rules_df[rules_df["antecedents"].apply(lambda a: a.issubset(items))]
    recs = []
    for _, r in cand.sort_values(["lift", "confidence"], ascending=False).iterrows():
        for c in r["consequents"]:
            if c not in items and c not in recs:
                recs.append(c)
        if len(recs) >= n:
            break
    return recs[:n]


# examples: take the first antecedent of the top rules
for ant in rules["antecedents"].head(3):
    print(f"\nCustomer bought: {sorted(ant)}")
    print("Recommended     :", recommend(ant))

# %% E. Check the recommender on the held-out invoices (hit rate)
rng = np.random.default_rng(RANDOM_STATE)
sample = test_b.sample(n=min(1000, len(test_b)), random_state=RANDOM_STATE)
hits, total = 0, 0
for _, row in sample.iterrows():
    items = row.index[row.values].tolist()
    if len(items) < 2:
        continue
    hidden = items[rng.integers(len(items))]
    given = [i for i in items if i != hidden]
    recs = recommend(given, n=5)
    if recs:
        total += 1
        hits += hidden in recs
print(f"\nHit-rate@5 on test invoices (where a recommendation exists): "
      f"{hits}/{total} = {hits / max(total, 1):.2%}")
