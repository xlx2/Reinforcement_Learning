import gymnasium as gym
import numpy as np


class FluidAntennaSystemEnv(gym.Env):
    def __init__(self, parameters: dict):
        super().__init__()
        self.num_of_antennas = parameters["num_of_antennas"]
        self.num_of_users = parameters["num_of_users"]

        self.action_space = gym.spaces.MultiBinary((
            self.num_of_antennas, self.num_of_users + self.num_of_antennas
        ))
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(parameters["channel"].size * 2,),
                dtype=np.float32
            ),
            gym.spaces.Box(
                low=-90,
                high=90,
                shape=(parameters["doa"].size,),
                dtype=np.float32
            )
        ))

    def step(self, action) -> tuple[tuple[np.ndarray, np.ndarray], np.float32, bool, bool, {}]:

        return observations, rewards, terminated, truncated, {}

    def reset(self, seed, options) -> tuple[tuple[np.ndarray, np.ndarray], {}]:

        return observations, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass
