import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def load_any(name):
    """Load either CSV or XLSX by filename stem."""
    stem = os.path.join(DATA_DIR, name)
    if os.path.exists(stem + ".csv"):
        return pd.read_csv(stem + ".csv", encoding="utf-8")
    if os.path.exists(stem + ".xlsx"):
        return pd.read_excel(stem + ".xlsx")
    raise FileNotFoundError(f"Neither {name}.csv nor {name}.xlsx found in data/")

books      = load_any("books")
ratings    = load_any("ratings")
to_read    = load_any("to_read")
book_tags  = load_any("book_tags")
tags       = load_any("tags")

print("✅ Loaded datasets")
print("books      :", books.shape)
print("ratings    :", ratings.shape)
print("to_read    :", to_read.shape)
print("book_tags  :", book_tags.shape)
print("tags       :", tags.shape)

# quick sanity: expected columns
print("\nColumns:")
for n, df in [("books", books), ("ratings", ratings), ("to_read", to_read), ("book_tags", book_tags), ("tags", tags)]:
    print(f"- {n}: {list(df.columns)}")

# small previews
print("\nHead:")
print("books.head():")
print(books.head(3).to_string(index=False))
print("\nratings.head():")
print(ratings.head(3).to_string(index=False))

# basic stats
n_users  = ratings["user_id"].nunique()
n_items  = ratings["book_id"].nunique()
n_inter  = len(ratings)
print(f"\nUsers: {n_users:,} | Items: {n_items:,} | Interactions: {n_inter:,}")
