# warp_synth_data.py
import time
import numpy as np
import gymnasium as gym
import warp as wp

ENV_ID = "InvertedPendulum-v5"

@wp.kernel
def add_noise(x: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32), sigma: float, seed: int):
    i = wp.tid()

    s = wp.uint32(seed) ^ wp.uint32(i * 747796405)

    s = s ^ (s << wp.uint32(13))
    s = s ^ (s >> wp.uint32(17))
    s = s ^ (s << wp.uint32(5))

    u = wp.float32(s % wp.uint32(10000)) / wp.float32(10000.0)  # [0,1)
    noise = (u - wp.float32(0.5)) * wp.float32(2.0) * wp.float32(sigma)

    y[i] = x[i] + noise

def collect_obs(n_steps=200_000):
    env = gym.make(ENV_ID)
    obs_list = []
    obs, _ = env.reset()
    for _ in range(n_steps):
        # random policy just for dataset; swap in trained policy later if needed
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        obs_list.append(obs.copy())
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    X = np.asarray(obs_list, dtype=np.float32)
    return X

if __name__ == "__main__":
    X = collect_obs(n_steps=100_000)  # 100k+ rows
    flat = X.reshape(-1).astype(np.float32)

    # Warp version
    wp.init()
    x_wp = wp.array(flat, dtype=wp.float32)
    y_wp = wp.empty_like(x_wp)

    t0 = time.time()
    wp.launch(add_noise, dim=flat.size, inputs=[x_wp, y_wp, 0.02, 123])
    wp.synchronize()
    warp_s = time.time() - t0

    Y_warp = y_wp.numpy().reshape(X.shape)

    # NumPy baseline
    t1 = time.time()
    Y_np = X + (np.random.rand(*X.shape).astype(np.float32) - 0.5) * 2.0 * 0.02
    numpy_s = time.time() - t1

    np.savez("synthetic_dataset.npz", X=X, Y_warp=Y_warp, Y_np=Y_np)
    print(f"Saved synthetic_dataset.npz with {X.shape[0]:,} rows")
    print(f"Warp time:  {warp_s:.4f}s")
    print(f"NumPy time: {numpy_s:.4f}s")
