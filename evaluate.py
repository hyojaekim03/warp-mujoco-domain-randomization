# evaluate.py
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from domain_randomization import DomainRandomizationWrapper

ENV_ID = "InvertedPendulum-v5"

def eval_model(env, model, episodes=20):
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += float(reward)
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns))

if __name__ == "__main__":
    model = PPO.load("ppo_inverted_pendulum_dr.zip")

    clean_env = gym.make(ENV_ID)
    rand_env = DomainRandomizationWrapper(gym.make(ENV_ID), seed=1)

    clean_mean, clean_std = eval_model(clean_env, model)
    rand_mean, rand_std = eval_model(rand_env, model)

    print(f"Clean sim return: {clean_mean:.1f} ± {clean_std:.1f}")
    print(f"Randomized sim return: {rand_mean:.1f} ± {rand_std:.1f}")

    clean_env.close()
    rand_env.close()
