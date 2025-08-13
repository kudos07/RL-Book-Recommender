import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ART_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ART_DIR, exist_ok=True)

# -------- 1) Load raw --------
def load_any(name):
    stem = os.path.join(DATA_DIR, name)
    if os.path.exists(stem + ".csv"):
        return pd.read_csv(stem + ".csv", encoding="utf-8")
    if os.path.exists(stem + ".xlsx"):
        return pd.read_excel(stem + ".xlsx")
    raise FileNotFoundError(f"{name} not found")

books     = load_any("books")
ratings   = load_any("ratings")
book_tags = load_any("book_tags")
tags      = load_any("tags")

# -------- 2) Implicit feedback --------
# Keep ratings >= 4 as positives
pos = ratings[ratings["rating"] >= 4][["user_id", "book_id"]].drop_duplicates()

# Filter out extremely cold users/items
user_counts = pos["user_id"].value_counts()
item_counts = pos["book_id"].value_counts()
pos = pos[pos["user_id"].isin(user_counts[user_counts >= 5].index)]
pos = pos[pos["book_id"].isin(item_counts[item_counts >= 10].index)]

# -------- 3) Build tag matrix for items --------
# book_tags: (goodreads_book_id, tag_id, count)
# tags: (tag_id, tag_name)
bt = book_tags.merge(tags, on="tag_id", how="left")

# Map item ids to a compact index
item_ids = sorted(pos["book_id"].unique())
item_id2idx = {bid: i for i, bid in enumerate(item_ids)}
idx2item_id = {i: bid for bid, i in item_id2idx.items()}

# Keep only tags for our filtered items
bt = bt[bt["goodreads_book_id"].isin(item_ids)].copy()
bt["item_idx"] = bt["goodreads_book_id"].map(item_id2idx)

# Build a sparse matrix: rows = items, cols = tags
tag_ids = sorted(bt["tag_id"].unique())
tag_id2idx = {tid: j for j, tid in enumerate(tag_ids)}
bt["tag_idx"] = bt["tag_id"].map(tag_id2idx)

rows = bt["item_idx"].to_numpy()
cols = bt["tag_idx"].to_numpy()
vals = bt["count"].to_numpy(dtype=np.float32)
n_items = len(item_ids)
n_tags = len(tag_ids)
M = csr_matrix((vals, (rows, cols)), shape=(n_items, n_tags))

# TF-IDF normalize, then SVD to 64 dims for compact item vectors
tfidf = TfidfTransformer()
M_tfidf = tfidf.fit_transform(M)

svd = TruncatedSVD(n_components=min(64, max(8, int(min(M_tfidf.shape) * 0.5))))
item_vecs = svd.fit_transform(M_tfidf).astype(np.float32)  # shape (n_items, d)
d = item_vecs.shape[1]

# -------- 4) Popularity & fixed action space --------
pop = pos["book_id"].value_counts()
top_items = list(pop.index[:200]) if len(pop) >= 200 else list(pop.index)
# Ensure these items are in our filtered item list
top_items = [bid for bid in top_items if bid in item_id2idx]

action_item_indices = np.array([item_id2idx[bid] for bid in top_items], dtype=np.int64)
n_actions = len(action_item_indices)

# -------- 5) Per-user split: history vs online --------
# Random 80/20 split per user (no timestamp available in goodbooks-10k)
rng = np.random.default_rng(42)
hist_pos = []
online_pos = []
for uid, grp in pos.groupby("user_id"):
    items = grp["book_id"].to_numpy()
    if len(items) < 5:
        continue
    mask = rng.random(len(items)) < 0.8
    hist = items[mask]
    onln = items[~mask]
    # guard if all went to one side
    if len(onln) == 0 and len(hist) > 1:
        onln = hist[-1:]
        hist = hist[:-1]
    if len(hist) == 0:
        continue
    hist_pos.append(pd.DataFrame({"user_id": uid, "book_id": hist}))
    if len(onln) > 0:
        online_pos.append(pd.DataFrame({"user_id": uid, "book_id": onln}))

hist_pos = pd.concat(hist_pos, ignore_index=True)
online_pos = pd.concat(online_pos, ignore_index=True) if len(online_pos) else pd.DataFrame(columns=["user_id","book_id"])

# Keep users present in history
valid_users = sorted(hist_pos["user_id"].unique())
user_id2idx = {uid: i for i, uid in enumerate(valid_users)}
idx2user_id = {i: uid for uid, i in user_id2idx.items()}

# Map to indices
hist_pos["u_idx"] = hist_pos["user_id"].map(user_id2idx)
hist_pos["i_idx"] = hist_pos["book_id"].map(item_id2idx)
hist_pos = hist_pos.dropna().astype({"u_idx": int, "i_idx": int})

if len(online_pos) > 0:
    online_pos = online_pos[online_pos["user_id"].isin(valid_users)]
    online_pos["u_idx"] = online_pos["user_id"].map(user_id2idx)
    online_pos["i_idx"] = online_pos["book_id"].map(item_id2idx)
    online_pos = online_pos.dropna().astype({"u_idx": int, "i_idx": int})

# -------- 6) Build user vectors from history --------
# user_vec = mean of item_vecs for their history positives
user_vecs = np.zeros((len(valid_users), d), dtype=np.float32)
for u, grp in hist_pos.groupby("u_idx"):
    iv = grp["i_idx"].to_numpy()
    user_vecs[u] = item_vecs[iv].mean(axis=0)

# -------- 7) Build fast lookup sets for online positives (for reward)
from collections import defaultdict
online_sets = defaultdict(set)
for _, r in online_pos.iterrows():
    online_sets[int(r["u_idx"])].add(int(r["i_idx"]))

# -------- 8) Save artifacts --------
np.save(os.path.join(ART_DIR, "item_vecs.npy"), item_vecs)
np.save(os.path.join(ART_DIR, "user_vecs.npy"), user_vecs)
np.save(os.path.join(ART_DIR, "action_item_indices.npy"), action_item_indices)

with open(os.path.join(ART_DIR, "mappings.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "d": int(d),
            "n_items": int(n_items),
            "n_users": int(len(valid_users)),
            "n_actions": int(n_actions),
        },
        f,
    )

# Save online positive lists as a compact npz
# arrays: users[], offsets[], items[] to reconstruct per-user sets quickly
users_sorted = sorted(online_sets.keys())
offsets = [0]
items_flat = []
for u in users_sorted:
    s = sorted(list(online_sets[u]))
    items_flat.extend(s)
    offsets.append(len(items_flat))
np.savez(os.path.join(ART_DIR, "online_sets.npz"),
         users=np.array(users_sorted, dtype=np.int32),
         offsets=np.array(offsets, dtype=np.int32),
         items=np.array(items_flat, dtype=np.int32))

print("Preprocess complete")
print("Users:", len(valid_users), "| Items:", n_items, "| Action items:", n_actions, "| Dim:", d)
