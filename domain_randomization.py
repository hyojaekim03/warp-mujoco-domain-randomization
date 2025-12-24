# domain_randomization.py
import numpy as np
import gymnasium as gym

class DomainRandomizationWrapper(gym.Wrapper):
    """
    Randomizes MuJoCo model parameters on every reset (sim-to-real style).
    Works with Gymnasium MuJoCo envs that expose env.unwrapped.model.
    """

    def __init__(self, env: gym.Env, seed: int = 0,
                 mass_scale: float = 0.15,
                 friction_scale: float = 0.2,
                 damping_scale: float = 0.2):
        super().__init__(env)
        self.rng = np.random.default_rng(seed)

        self.mass_scale = mass_scale
        self.friction_scale = friction_scale
        self.damping_scale = damping_scale

        model = self.env.unwrapped.model
        # Save originals so each episode starts from a known baseline
        self._orig_body_mass = model.body_mass.copy()
        self._orig_geom_friction = model.geom_friction.copy()
        self._orig_dof_damping = model.dof_damping.copy()

    def _apply_randomization(self):
        m = self.env.unwrapped.model

        # Restore baseline first
        m.body_mass[:] = self._orig_body_mass
        m.geom_friction[:] = self._orig_geom_friction
        m.dof_damping[:] = self._orig_dof_damping

        # Apply multiplicative noise (clipped to keep sim stable)
        mass_mult = self.rng.uniform(1 - self.mass_scale, 1 + self.mass_scale, size=m.body_mass.shape)
        m.body_mass[:] = np.clip(m.body_mass * mass_mult, 1e-6, None)

        fric_mult = self.rng.uniform(1 - self.friction_scale, 1 + self.friction_scale, size=m.geom_friction.shape)
        m.geom_friction[:] = np.clip(m.geom_friction * fric_mult, 1e-6, None)

        damp_mult = self.rng.uniform(1 - self.damping_scale, 1 + self.damping_scale, size=m.dof_damping.shape)
        m.dof_damping[:] = np.clip(m.dof_damping * damp_mult, 1e-6, None)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._apply_randomization()
        return obs, info
