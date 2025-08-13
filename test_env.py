import os
import numpy as np
from envs.goodbooks_env import make_env

ART = os.path.join(os.path.dirname(__file__), "artifacts")
env = make_env(ART, episode_len=5, seed=123)

n_episodes = 10
total_r = 0.0
hits = 0
steps = 0

for ep in range(n_episodes):
    obs, info = env.reset()
    done = False
    trunc = False
    while not (done or trunc):
        a = env.action_space.sample()  # random action
        obs, r, done, trunc, info = env.step(a)
        total_r += r
        hits += int(r > 0)
        steps += 1

print(f"Episodes: {n_episodes}, Steps: {steps}, Random total reward: {total_r:.1f}, hits: {hits}")
print("Obs dim:", obs.shape)
