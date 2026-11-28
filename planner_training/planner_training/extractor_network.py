import gymnasium as gym
import torch as th
import numpy as np

from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class NavFeatExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim = 64)

        # Determine subspace sizes
        self.state_size = 0
        for key, subspace in observation_space.spaces.items():
            if key == 'scan':
                self.scan_size = subspace.shape[0]
            else:
                self.state_size += subspace.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=5,
                stride=2
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=2
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=32,
                out_channels=1,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=26, out_features=self.scan_size),
            nn.ReLU()
        )

        self.lnn = nn.Sequential(
            nn.Linear(
                in_features=self.scan_size + self.state_size,
                out_features=self.scan_size
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=self.scan_size,
                out_features=self.scan_size
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=self.scan_size,
                out_features=64
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=64,
                out_features=64
            ),
            nn.ReLU()
        )

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        scan = observations["scan"].unsqueeze(1)
        cnn_feats = self.cnn(scan)

        state = th.cat(
            (
                observations["heading"],
                observations["distance"],
                observations["velocity"],
            ),
            dim=1,
        )

        x = th.cat((cnn_feats, state), dim=1)
        return self.lnn(x)