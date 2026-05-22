from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces


DEFAULT_XML = str(Path(__file__).parent / "assets" / "continuum_robot.xml")


_TARGET_POOL_CACHE: dict[str, np.ndarray] = {}


def _build_target_pool(
    xml_path: str,
    n_samples: int = 300,
    settle_steps: int = 1500,
    seed: int = 42,
) -> np.ndarray:
    """Pre-sample tip positions that are reachable by some constant control.

    Random target sampling in a 3D cone produces mostly unreachable points
    because the tip is confined to a 2D manifold parameterised by the two
    tendon commands. Instead we sweep random controls, let the dynamics
    settle, and record the resulting tip positions. Targets drawn from this
    pool are reachable by construction.
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")
    rng = np.random.default_rng(seed)
    pool = np.zeros((n_samples, 3), dtype=np.float64)
    for i in range(n_samples):
        mujoco.mj_resetData(model, data)
        ctrl = rng.uniform(-1.0, 1.0, size=model.nu)
        ctrl *= rng.uniform(0.2, 1.0)  # bias toward interior of action space
        data.ctrl[:] = ctrl
        for _ in range(settle_steps):
            mujoco.mj_step(model, data)
        pool[i] = data.site_xpos[tip_id]
    return pool


def get_target_pool(xml_path: str) -> np.ndarray:
    pool = _TARGET_POOL_CACHE.get(xml_path)
    if pool is None:
        pool = _build_target_pool(xml_path)
        _TARGET_POOL_CACHE[xml_path] = pool
    return pool


class ContinuumReachEnv(gym.Env):
    """Tip-reaching task for a tendon-driven continuum robot.

    The robot is a piecewise-constant-curvature approximation built from 8
    short rigid links connected by 2-DOF universal joints with passive
    stiffness and damping. Two fixed tendons bundle the x- and y-axis hinges
    so that a single motor per axis drives the whole bundle, mimicking a
    pair of antagonistic tendons running through the catheter.

    Action: ctrl in [-1, 1]^2 (bend_x, bend_y).
    Reward: dense negative tip-to-target distance, small control penalty,
            and a +1 bonus when the tip is within ``success_dist``.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    def __init__(
        self,
        xml_path: str = DEFAULT_XML,
        n_substeps: int = 10,
        max_steps: int = 200,
        success_dist: float = 0.005,
        render_mode: str | None = None,
    ):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.n_substeps = n_substeps
        self.max_steps = max_steps
        self.success_dist = success_dist
        self.render_mode = render_mode
        self._step_count = 0
        self._renderer: mujoco.Renderer | None = None

        self.tip_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "tip"
        )
        if self.tip_site_id < 0:
            raise RuntimeError("MJCF is missing a 'tip' site")
        self.target = np.zeros(3, dtype=np.float64)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )
        obs_dim = self.model.nq + self.model.nv + 9
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._target_pool = get_target_pool(xml_path)

    def _sample_target(self, rng: np.random.Generator) -> np.ndarray:
        idx = int(rng.integers(0, len(self._target_pool)))
        return self._target_pool[idx].copy()

    def _tip(self) -> np.ndarray:
        return self.data.site_xpos[self.tip_site_id].copy()

    def _obs(self) -> np.ndarray:
        tip = self._tip()
        return np.concatenate(
            [
                self.data.qpos,
                self.data.qvel,
                tip,
                self.target,
                tip - self.target,
            ]
        ).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.target = self._sample_target(self.np_random)
        self._step_count = 0
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), {"target": self.target.copy()}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.data.ctrl[:] = action
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        tip = self._tip()
        dist = float(np.linalg.norm(tip - self.target))
        success = dist < self.success_dist
        reward = -dist - 0.001 * float(np.dot(action, action)) + (1.0 if success else 0.0)
        terminated = bool(success)
        truncated = self._step_count >= self.max_steps
        info = {"tip": tip, "dist": dist, "is_success": success}
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=240, width=320)
        self._renderer.update_scene(self.data, camera="side")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
