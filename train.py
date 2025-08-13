import os
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from envs.goodbooks_env import make_env

# ---- config ----
ART = os.path.join(os.path.dirname(__file__), "artifacts")
EP_LEN = 10
SEED = 42
TOTAL_STEPS = 200_000              # ~1–3 minutes on GPU; increase later
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# ---- env ----
env = make_env(ART, episode_len=EP_LEN, seed=SEED)
env = Monitor(env)                # records episode stats

# ---- sanity: GPU ----
print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))

# ---- PPO ----
policy_kwargs = dict(net_arch=[256, 256])
model = PPO(
    "MlpPolicy",
    env,
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=SEED,
    verbose=1,
    n_steps=512,          # was 128
    batch_size=256,       # factor of 512
    learning_rate=3e-4,
    gamma=0.95,
    gae_lambda=0.95,
    clip_range=0.2,
    n_epochs=10,
    policy_kwargs=dict(net_arch=[256, 256]),
)
print("Training…")
model.learn(total_timesteps=TOTAL_STEPS)

# ---- save model ----
SAVE_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(SAVE_DIR, exist_ok=True)
path = os.path.join(SAVE_DIR, "ppo_goodbooks.zip")
model.save(path)
print("Saved:", path)

# ---- quick eval vs random ----
from envs.goodbooks_env import make_env
import numpy as np

def eval_policy(env, model, episodes=100):
    hits, steps = 0, 0
    for _ in range(episodes):
        obs, info = env.reset()
        done = False; trunc = False
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, info = env.step(int(action))
            # count only ground-truth hits
            if info.get("is_hit", False):
                hits += 1
            steps += 1
    return hits / steps if steps else 0.0


def eval_random(env, episodes=100):
    hits, steps = 0, 0
    for _ in range(episodes):
        obs, info = env.reset()
        done = False; trunc = False
        while not (done or trunc):
            a = env.action_space.sample()
            obs, r, done, trunc, info = env.step(a)
            if info.get("is_hit", False):
                hits += 1
            steps += 1
    return hits / steps if steps else 0.0

eval_env = make_env(ART, episode_len=EP_LEN, seed=123)
rl_ctr = eval_policy(eval_env, model, episodes=200)
rand_ctr = eval_random(make_env(ART, episode_len=EP_LEN, seed=456), episodes=200)
print(f"Random true-hit CTR@1-per-step: {rand_ctr:.3f} | PPO true-hit CTR@1-per-step: {rl_ctr:.3f}")