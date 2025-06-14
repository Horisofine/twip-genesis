import gymnasium as gym
import numpy as np
import genesis as gs
from gymnasium import spaces

class TwipEnv(gym.Env):
    def __init__(self):
        super.__init__()

        #define the spaces
        self.observation_space = spaces.Box(low=-35, high=35, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

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

        self._joints_name = [
            "rwheel",
            "lwheel",
        ]
        self._motors_dof_idx = [self.twip.get_joint(name).dof_idx_local for name in self._joints_name]

        self.scene.build()

    def step(self, action):


        pass

    def reset(self):
        pass

    def close(self):
        pass