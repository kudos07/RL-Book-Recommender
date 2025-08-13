import os
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GoodbooksRecEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, artifacts_dir: str, episode_len: int = 5, seed: int = 42):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.episode_len = int(episode_len) 

        # Load artifacts
        self.item_vecs = np.load(os.path.join(artifacts_dir, "item_vecs.npy"))
        self.user_vecs = np.load(os.path.join(artifacts_dir, "user_vecs.npy"))
        self.action_item_indices = np.load(os.path.join(artifacts_dir, "action_item_indices.npy"))
        with open(os.path.join(artifacts_dir, "mappings.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.d = int(meta["d"])
        self.n_actions = int(meta["n_actions"])

        # Load online positives
        online = np.load(os.path.join(artifacts_dir, "online_sets.npz"))
        users = online["users"]
        offsets = online["offsets"]
        items = online["items"]
        self.online_sets = {}
        for k, u in enumerate(users):
            start, end = int(offsets[k]), int(offsets[k + 1])
            self.online_sets[int(u)] = set(int(i) for i in items[start:end])

        # Keep only valid users (at least 1 candidate hit)
        self.action_set = set(int(i) for i in self.action_item_indices.tolist())
        self.valid_users = [u for u, s in self.online_sets.items() if len(s.intersection(self.action_set)) > 0]
        assert self.valid_users, "No valid users found. Reduce candidate size or rebuild artifacts."

        # Spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.d,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_actions)

        # Reward shaping knobs
        self.hit_reward = 1.0
        self.sim_weight = 0.2
        self.novelty_bonus = 0.05
        self.repeat_penalty = 0.2
        self.diversity_bonus = 0.05
        self.diversity_window = 3
        self._user_norms = np.linalg.norm(self.user_vecs, axis=1) + 1e-8

        # State vars
        self._t = 0
        self._u = None
        self._user_vec = None
        self._seen_items = None
        self._recent_item_vecs = None

    def _sample_user(self):
        return int(self.valid_users[self.rng.integers(0, len(self.valid_users))])

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._t = 0
        self._u = self._sample_user()
        self._user_vec = self.user_vecs[self._u].astype(np.float32, copy=True)
        self._seen_items = set()
        self._recent_item_vecs = []
        return self._user_vec, {"user_idx": self._u}

    def step(self, action: int):
        assert self.action_space.contains(action)
        i_idx = int(self.action_item_indices[int(action)])
        item_vec = self.item_vecs[i_idx]
        user_vec = self._user_vec

        # cosine sim
        num = float(np.dot(user_vec, item_vec))
        denom = float(self._user_norms[self._u] * (np.linalg.norm(item_vec) + 1e-8))
        cos_sim = max(0.0, min(1.0, num / denom))

        reward = 0.0
        is_hit = (i_idx in self.online_sets.get(self._u, set()))
        is_repeat = (i_idx in self._seen_items)

        if is_hit and not is_repeat:
            reward += self.hit_reward
        reward += self.sim_weight * cos_sim
        if is_repeat:
            reward -= self.repeat_penalty
        else:
            reward += self.novelty_bonus

        if self._recent_item_vecs:
            sims = []
            ivn = np.linalg.norm(item_vec) + 1e-8
            for vprev in self._recent_item_vecs[-self.diversity_window:]:
                sims.append(float(np.dot(item_vec, vprev) / (ivn * (np.linalg.norm(vprev) + 1e-8))))
            avg_sim = sum(max(0.0, s) for s in sims) / len(sims)
            if avg_sim < 0.3:
                reward += self.diversity_bonus

        self._t += 1
        if not is_repeat:
            self._seen_items.add(i_idx)
            self._recent_item_vecs.append(item_vec)
            if len(self._recent_item_vecs) > self.diversity_window:
                self._recent_item_vecs = self._recent_item_vecs[-self.diversity_window:]

        return (
            user_vec,
            float(reward),
            False,
            self._t >= self.episode_len,
            {"is_hit": is_hit, "item_idx": i_idx}
        )

def make_env(artifacts_dir: str, episode_len: int = 5, seed: int = 42):
    return GoodbooksRecEnv(artifacts_dir=artifacts_dir, episode_len=episode_len, seed=seed)
