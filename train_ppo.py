# train_ppo.py
import gymnasium as gym
from stable_baselines3 import PPO
from domain_randomization import DomainRandomizationWrapper
from stable_baselines3.common.logger import configure


ENV_ID = "InvertedPendulum-v5"  # fast + simple MuJoCo control task

def make_env(randomize: bool):
    env = gym.make(ENV_ID)
    if randomize:
        env = DomainRandomizationWrapper(env, seed=0)
    return env

if __name__ == "__main__":
    env = make_env(randomize=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
    )

    new_logger = configure("logs/", ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    model.learn(total_timesteps=200_000)  # ~minutes, depending on machine
    model.save("ppo_inverted_pendulum_dr")

    env.close()
    print("Saved: ppo_inverted_pendulum_dr.zip")
