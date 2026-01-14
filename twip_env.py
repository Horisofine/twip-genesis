import math
import gymnasium as gym
import numpy as np
import torch
import genesis as gs
from gymnasium import spaces

class TwipEnv(gym.Env):
    def __init__(self):
        super(TwipEnv, self).__init__()

        self.device = torch.device("cpu")  # CUDA support

        self._max_torque = 10.0
        self._max_angle = 15.0

        # Define the spaces
        # Observation space
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.inf, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.pi, np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )

        self.obs = torch.zeros((4,), device=self.device)
        self.rew = torch.tensor(0.0, device=self.device)
        self.dones = False
        self.truncated = False

        # Action space
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.action = torch.zeros((2,), device=self.device)

        # Defining the scene
        gs.init(backend=gs.cpu)
        from genesis.engine.entities import RigidEntity

        self.scene = gs.Scene(
            vis_options=gs.options.VisOptions(
                show_link_frame=True,
            ),
            show_viewer=True
        )
        self.plane = self.scene.add_entity(gs.morphs.Plane())
        self.twip: RigidEntity = self.scene.add_entity(
            gs.morphs.URDF(file="assets/twip.urdf", pos=[0, 0, 0.1])
        )

        self._joints_name = ["lwheel", "rwheel"]
        self._motors_dof_idx = [self.twip.get_joint(name).dof_idx_local for name in self._joints_name]

        self.scene.build()

        self.twip.set_dofs_force_range(
            lower=-self._max_torque,
            upper=self._max_torque,
            dofs_idx_local=self._motors_dof_idx
        )

        self._init_pos = self.twip.get_pos().clone()
        self._init_quat = self.twip.get_quat().clone()

    def step(self, action):
        self.action = action * self._max_torque

        self.twip.control_dofs_force(self.action, self._motors_dof_idx)
        self.scene.step()

        self.dones = self._check_dones()
        self.truncated = False

        if self.dones:
            self.reset()

            self.obs = torch.zeros((4,), device=self.device)
            self.rew = torch.tensor(-1.0, device=self.device)
        else:
            self.obs = self._get_obs()
            self.rew = self._compute_rew()

        info = {
            "position": self.twip.get_pos(),
            "obs": self.obs,
            "rew": self.rew,
        }

        return self.obs, self.rew, self.dones, self.truncated, info

    def reset(self, seed=None, options=None):
        self.twip.set_dofs_position(
            position=torch.tensor([0, 0], dtype=torch.float32, device=self.device),
            dofs_idx_local=self._motors_dof_idx,
            zero_velocity=True
        )
        self.twip.set_pos(self._init_pos)
        self.twip.set_quat(self._init_quat)

        self.action = torch.zeros((2,), device=self.device)
        self.obs = torch.zeros((4,), device=self.device)

        return self.obs, None

    def close(self):
        pass

    def _get_obs(self):
        curr_yaw, _, _ = self._get_euler_ang()

        ang_vel = torch.tensor(self.twip.get_ang(), device=self.device)
        curr_ang_yaw = ang_vel[0]

        dofs_vel = torch.tensor(self.twip.get_dofs_velocity(self._motors_dof_idx), device=self.device)
        left_wheel_vel = dofs_vel[0]
        right_wheel_vel = dofs_vel[1]

        return torch.stack([
            curr_yaw,
            curr_ang_yaw,
            left_wheel_vel,
            right_wheel_vel
        ], dim=0)

    def _compute_rew(self):
        yaw, _, _ = self._get_euler_ang()
        yaw_threshold = torch.deg2rad(torch.tensor(self._max_angle, device=self.device))
        reward = 1.0 - (torch.abs(yaw) / yaw_threshold)
        return torch.clamp(reward, 0.0, 1.0)

    def _check_dones(self):
        yaw, _, _ = self._get_euler_ang()
        yaw_threshold = torch.deg2rad(torch.tensor(self._max_angle, device=self.device))
        return torch.abs(yaw) > yaw_threshold

    def _get_euler_ang(self):
        quat = torch.tensor(self.twip.get_quat(), device=self.device)
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (y * y + x * x)
        yaw = torch.atan2(t0, t1)

        t2 = 2.0 * (w * y - z * x)
        t2 = torch.clamp(t2, -1.0, 1.0)
        pitch = torch.asin(t2)

        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        roll = torch.atan2(t3, t4)

        return yaw, pitch, roll
