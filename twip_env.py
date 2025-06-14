import gymnasium as gym
import numpy as np
import genesis as gs
from gymnasium import spaces


class TwipEnv(gym.Env):
    def __init__(self):
        super.__init__()

        self.max_torque = 10.0

        # define the spaces
        # Observations are as follows: Pitch angle, Pitch angular velocity, left wheel velocity, right wheel velocity
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.inf, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.pi, np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )

        # Actions are the torques given to the motors
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        # Defining the scene
        gs.init(backend=gs.cuda)
        self.scene = gs.Scene()
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )
        self.twip = self.scene.add_entity(
            gs.morphs.URDF(
                file="assets/twip.urdf",
                pos=[0, 0, 0.1]
            )
        )

        # Defining the dofs
        self._joints_name = [
            "rwheel",
            "lwheel",
        ]
        self._motors_dof_idx = [self.twip.get_joint(name).dof_idx_local for name in self._joints_name]

        # Building the Scene
        self.scene.build()

    def step(self, action):
        # Apply the actions
        action = action.copy() * self.max_torque

        self.twip.control_dofs_force(
            action,
            self._motors_dof_idx,
        )

        # Step the sim
        self.scene.step()

        obs = self._get_obs()
        rew = self._compute_rew()
        dones = self._check_dones()

        info = {}

        return obs, rew, dones, info

    def reset(self):
        pass

    def close(self):
        pass

    def _get_obs(self):
        pass

    def _compute_rew(self):
        pass

    def _check_dones(self):
        pass