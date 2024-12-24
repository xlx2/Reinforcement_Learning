from utils import dB2pow
from fluid_antenna_system import FluidAntennaSystem
import numpy as np

np.random.seed(100)  # fix the random parameters

# Initialization
NUM_OF_USERS = 5
NUM_OF_ANTENNAS = 16
NUM_OF_SELECTED_ANTENNAS = 4
CHANNEL_NOISE = dB2pow(-10)
SENSING_NOISE = dB2pow(-10)
REFLECTION_COEFFICIENT = dB2pow(-10)
QOS_THRESHOLD = dB2pow(12)
SENSING_UPPER_THRESHOLD = dB2pow(12)
SENSING_LOWER_THRESHOLD = dB2pow(8)
DOA = np.array([[np.pi / 6, 0, -np.pi/6]])
indices = np.arange(0, (NUM_OF_ANTENNAS - 1) + 1).reshape(-1, 1)
STEERING_VECTOR = np.exp(-1j * np.pi * indices * np.sin(DOA))

fas = FluidAntennaSystem(num_of_yaxis_antennas=NUM_OF_ANTENNAS, num_of_users=NUM_OF_USERS, noise_variance=CHANNEL_NOISE)
CHANNEL, _, _, _ = fas.get_channel()

Parameters = {"num_of_antennas": NUM_OF_ANTENNAS,
              "num_of_users": NUM_OF_USERS,
              "num_of_selected_antennas": NUM_OF_SELECTED_ANTENNAS,
              "qos_threshold": QOS_THRESHOLD,
              "sensing_upper_threshold": SENSING_UPPER_THRESHOLD,
              "sensing_lower_threshold": SENSING_LOWER_THRESHOLD,
              "reflection_coefficient": REFLECTION_COEFFICIENT,
              "channel_noise": CHANNEL_NOISE,
              "sensing_noise": SENSING_NOISE,
              "doa": DOA,
              "steering_vector": STEERING_VECTOR,
              "channel": CHANNEL}
