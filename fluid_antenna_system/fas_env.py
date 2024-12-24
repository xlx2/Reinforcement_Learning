import gymnasium as gym
import numpy as np


class FluidAntennaSystemEnv(gym.Env):
    def __init__(self, parameters: dict):
        super().__init__()
        self.num_of_selected_antennas = parameters["num_of_selected_antennas"]
        self.num_of_antennas = parameters["num_of_antennas"]
        self.num_of_users = parameters["num_of_users"]

        self.action_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.num_of_antennas, self.num_of_users),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(parameters["channel"].shape[0], parameters["channel"].shape[1], 2),
            dtype=np.float32
        )

    def step(self, action) -> tuple[np.ndarray, np.float32, bool, bool, {}]:

        return observations, rewards, terminated, truncated, {}

    def reset(self,) -> tuple[np.ndarray, {}]:

        return observations, {}