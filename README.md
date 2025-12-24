# 🤖 Warp + MuJoCo Sim-to-Real (Domain Randomization) Demo

Minimal robotics RL project that:
- trains a MuJoCo control policy with **domain randomization** (sim-to-real style)
- evaluates the trained policy in **clean vs randomized** simulation
- generates a **100K+ row synthetic dataset** and applies fast observation noise using a **Warp kernel**

> Note: On Apple Silicon, Warp runs on **CPU** (CUDA is not available). The project still demonstrates Warp kernel programming and accelerated, parallel data transforms.

---

## Project Structure

├── domain_randomization.py # Wrapper that randomizes MuJoCo model params each reset

├── train_ppo.py # PPO training (Stable-Baselines3) with domain randomization

├── evaluate.py # Clean vs randomized evaluation

├── warp_synth_data.py # 100K+ synthetic obs + Warp kernel noise transform

├── ppo_inverted_pendulum_dr.zip # Saved SB3 model checkpoint (auto-generated)

└── logs/

├── progress.csv # SB3 CSV logs (if enabled)

└── events.out.tfevents... # TensorBoard logs (if enabled)


