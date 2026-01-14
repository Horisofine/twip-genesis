import math
from typing import Any

import genesis as gs
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices

from randomization import random_range
from utils import rotate_quat_pitch, quat_to_rpy


class TwipEnv(VecEnv):
    def __init__(self, num_envs: int = 8):
        observation_space = spaces.Box(
            low=np.array([-np.pi, -np.inf, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.pi, np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )
        action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        super(TwipEnv, self).__init__(
            num_envs=num_envs,
            observation_space=observation_space,
            action_space=action_space,
        )

        self.render_mode = "human"
        self.device = torch.device("cpu")  # CUDA support

        self._max_torque = 10.0 # Nm
        self._touchdown_roll = math.radians(14.75)
        self._max_roll = math.radians(17.0)
        self._max_episode_length = 512  # steps

        self.obs = torch.zeros((self.num_envs, 4), device=self.device)
        self.actions = torch.zeros((self.num_envs, 2), device=self.device)
        self.rew = torch.zeros((self.num_envs,), device=self.device)
        self.lengths = torch.zeros((self.num_envs,), device=self.device)
        self.dones = False
        self.truncated = False

        # Defining the scene
        gs.init(backend=gs.cpu)
        from genesis.engine.entities import RigidEntity

        max_rendered_num_envs = 3
        self.scene = gs.Scene(
            vis_options=gs.options.VisOptions(
                show_link_frame=False,
                rendered_envs_idx=torch.randperm(max_rendered_num_envs).tolist(),
            ),
            viewer_options=gs.options.ViewerOptions(
                res=(1200, 800),
            ),
            show_viewer=True,
            show_FPS=False,
        )
        self.plane = self.scene.add_entity(gs.morphs.Plane())
        self.twip: RigidEntity = self.scene.add_entity(
            gs.morphs.URDF(file="assets/twip.urdf", pos=[0, 0, 0.1])
        )

        self._joints_name = ["lwheel", "rwheel"]
        self._motors_dof_idx = [self.twip.get_joint(name).dof_idx_local for name in self._joints_name]

        self.scene.build(
            n_envs=self.num_envs,
        )

        self.twip.set_dofs_force_range(
            lower=-self._max_torque,
            upper=self._max_torque,
            dofs_idx_local=self._motors_dof_idx,
        )

        self._init_pos = self.twip.get_pos().clone()
        self._init_quat = self.twip.get_quat().clone()

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = torch.tensor(actions, device=self.device) * self._max_torque

    def step_wait(self):
        self.twip.control_dofs_force(self.actions, self._motors_dof_idx)
        self.scene.step()

        self.dones = self._check_dones()
        self.truncated = self.lengths >= self._max_episode_length

        envs_to_reset = self.dones | self.truncated

        # create the env idx tensor for all environments that are done (i.e. are True)
        done_envs_idx = torch.nonzero(envs_to_reset, as_tuple=False).squeeze(-1)

        self.obs = self._get_obs()
        self.rew = self._compute_rew()
        self.lengths += 1

        if done_envs_idx.numel() > 0:
            reset_obs, reset_action, reset_rew = self._reset_envs(envs_idx=done_envs_idx)

            self.obs[done_envs_idx] = reset_obs
            self.actions[done_envs_idx] = reset_action
            self.rew[done_envs_idx] = reset_rew
            self.lengths[done_envs_idx] = 0

        infos = []
        for i in range(self.num_envs):
            infos.append({
                "position": self.twip.get_pos(i),  # per-env
                "obs": self.obs[i].cpu().numpy(),
                "action": self.actions[i].cpu().numpy(),
                "rew": float(self.rew[i]),
                "done": bool(self.dones[i]),
                "truncated": bool(self.truncated[i]),
                "length": int(self.lengths[i]),
            })

        return self.obs.numpy(), self.rew.numpy(), self.dones.numpy(), infos

    def reset(self, seed=None, options=None, envs_idx: torch.Tensor | None = None):
        if envs_idx is None:
            envs_idx = torch.arange(self.num_envs, device=self.device)

        self.obs[envs_idx], self.actions[envs_idx], self.rew[envs_idx] = self._reset_envs(envs_idx=envs_idx)

        return self.obs.numpy()

    def close(self):
        pass

    def _reset_envs(self, envs_idx: torch.Tensor):
        self.twip.set_dofs_position(
            position=torch.tensor([0, 0], dtype=torch.float32, device=self.device),
            dofs_idx_local=self._motors_dof_idx,
            envs_idx=envs_idx,
            zero_velocity=True
        )

        num_reset = len(envs_idx)

        # Position reset
        self.twip.set_pos(
            self._init_pos[envs_idx],
            envs_idx=envs_idx,
        )

        # Orientation reset
        max_roll = torch.ones((num_reset,), dtype=torch.float32, device=self.device) * self._max_roll
        random_rolls = random_range(-max_roll, max_roll)
        reset_quats = rotate_quat_pitch(self._init_quat[envs_idx], random_rolls)

        self.twip.set_quat(
            reset_quats,
            envs_idx=envs_idx,
        )

        action = torch.zeros((num_reset, 2), device=self.device)
        obs = torch.zeros((num_reset, 4), device=self.device)
        rew = -torch.ones((num_reset,), device=self.device)

        return obs, action, rew

    def _get_obs(self):
        curr_roll, _, _ = self._get_rpy()

        ang_vel = torch.tensor(self.twip.get_ang(), device=self.device)
        curr_ang_roll = ang_vel[:, 0]

        dofs_vel = torch.tensor(self.twip.get_dofs_velocity(self._motors_dof_idx), device=self.device)
        left_wheel_vel = dofs_vel[:, 0]
        right_wheel_vel = dofs_vel[:, 1]

        return torch.stack([
            curr_roll,
            curr_ang_roll,
            left_wheel_vel,
            right_wheel_vel
        ], dim=1)

    def _compute_rew(self):
        roll, _, _ = self._get_rpy()
        roll_threshold = torch.tensor(self._touchdown_roll, device=self.device)
        reward = 1.0 - (torch.abs(roll) / roll_threshold)
        return torch.clamp(reward, 0.0, 1.0)

    def _check_dones(self):
        roll, _, _ = self._get_rpy()
        roll_threshold = torch.tensor(self._max_roll, device=self.device)
        return torch.abs(roll) > roll_threshold

    def _get_rpy(self):
        quat = torch.tensor(self.twip.get_quat(), device=self.device)
        rpy = quat_to_rpy(quat)

        return rpy[:, 0], rpy[:, 1], rpy[:, 2]

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        return [getattr(self, attr_name) for _ in range(self.num_envs)]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list[Any]:
        return [getattr(self, method_name)(*method_args, **method_kwargs) for _ in range(self.num_envs)]

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None) -> list[bool]:
        return [isinstance(self, wrapper_class) for _ in range(self.num_envs)]
