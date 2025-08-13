# eval.py
import os
import numpy as np
import pandas as pd
from typing import List, Tuple
import gymnasium as gym
from stable_baselines3 import PPO

from envs.goodbooks_env import make_env

# ----- config -----
ROOT = os.path.dirname(__file__)
ART = os.path.join(ROOT, "artifacts")
MODEL_PATH = os.path.join(ROOT, "models", "ppo_goodbooks.zip")

EPISODES = 1000        # increase if you want tighter confidence
EP_LEN = 10            # must match what you trained with
TOPK_LIST = [1, 5, 10]
SEED = 123


# ----- metrics helpers -----
def ndcg_at_k(relevances: List[int], k: int) -> float:
    rel = np.array(relevances[:k], dtype=float)
    if rel.sum() == 0:
        return 0.0
    denom = np.log2(np.arange(2, len(rel) + 2))
    dcg = (rel / denom).sum()
    ideal = (np.sort(rel)[::-1] / denom).sum()
    return float(dcg / ideal) if ideal > 0 else 0.0


def precision_recall_at_k(relevances: List[int], k: int, positives_in_session: int) -> Tuple[float, float]:
    rel = np.array(relevances[:k], dtype=float)
    prec = rel.mean()
    rec = rel.sum() / max(positives_in_session, 1)
    return float(prec), float(rec)


# ----- evaluation core -----
def run_policy(env_factory, policy_fn, episodes=100, topk_list=(1, 5, 10)):
    rng = np.random.default_rng(SEED)

    ctr_hits_at_k = {k: 0 for k in topk_list}  # “at least one true hit in top‑k” per episode
    prec_sum = {k: 0.0 for k in topk_list}
    rec_sum = {k: 0.0 for k in topk_list}
    ndcg_sum = {k: 0.0 for k in topk_list}

    for _ in range(episodes):
        env = env_factory()
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))

        u = info.get("user_idx")
        # positives available inside candidate set (upper bound for recall)
        positives = len(env.online_sets.get(u, set()).intersection(set(env.action_item_indices.tolist())))

        relevances = []
        done = False
        trunc = False
        while not (done or trunc):
            a = policy_fn(env, obs)
            obs, r, done, trunc, info = env.step(int(a))
            # count ONLY true hits (not shaped reward):
            relevances.append(1 if info.get("is_hit", False) else 0)

        for k in topk_list:
            ctr_hits_at_k[k] += int(any(relevances[:k]))
            p, r = precision_recall_at_k(relevances, k, positives)
            n = ndcg_at_k(relevances, k)
            prec_sum[k] += p
            rec_sum[k] += r
            ndcg_sum[k] += n

        env.close()

    rows = []
    for k in topk_list:
        rows.append({"metric": "CTR@k_episode", "k": k, "value": round(ctr_hits_at_k[k] / episodes, 4)})
        rows.append({"metric": "Precision@k", "k": k, "value": round(prec_sum[k] / episodes, 4)})
        rows.append({"metric": "Recall@k", "k": k, "value": round(rec_sum[k] / episodes, 4)})
        rows.append({"metric": "NDCG@k", "k": k, "value": round(ndcg_sum[k] / episodes, 4)})
    return pd.DataFrame(rows)


# ----- baselines -----
def policy_random(env, obs):
    return env.action_space.sample()


def make_popularity_policy():
    # Recommend the first unseen action each step (assumes action list roughly popularity‑sorted)
    def _policy(env, obs):
        for a in range(env.n_actions):
            i_idx = int(env.action_item_indices[a])
            if i_idx not in env._seen_items:
                return a
        return env.action_space.sample()
    return _policy


if __name__ == "__main__":
    # new env instance per episode
    def env_factory():
        return make_env(ART, episode_len=EP_LEN, seed=SEED)

    print("Evaluating Random…")
    df_random = run_policy(env_factory, policy_random, episodes=EPISODES, topk_list=TOPK_LIST)
    df_random["model"] = "Random"

    print("Evaluating Popularity…")
    pop_policy = make_popularity_policy()
    df_pop = run_policy(env_factory, pop_policy, episodes=EPISODES, topk_list=TOPK_LIST)
    df_pop["model"] = "Popularity"

    print("Evaluating PPO…")
    assert os.path.exists(MODEL_PATH), f"Model not found at {MODEL_PATH}. Run train.py first."
    ppo = PPO.load(MODEL_PATH, device="cuda")
    def policy_ppo(env, obs):
        action, _ = ppo.predict(obs, deterministic=True)
        return int(action)
    df_ppo = run_policy(env_factory, policy_ppo, episodes=EPISODES, topk_list=TOPK_LIST)
    df_ppo["model"] = "PPO"

    df = pd.concat([df_random, df_pop, df_ppo], ignore_index=True)
    out_csv = os.path.join(ROOT, "eval_results.csv")
    df.to_csv(out_csv, index=False)
    print("\nSaved:", out_csv)

    pivot = df.pivot(index=["metric", "k"], columns="model", values="value").sort_index()
    print("\nResults:")
    print(pivot.round(4))
