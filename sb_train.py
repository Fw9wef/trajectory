import gym
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict
from env import Drones
import torch
from torch import nn
import torch.nn.functional as F


class FE(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim=32):
        super(FE, self).__init__(observation_space, features_dim)
        embed_dim = 16
        nhead = 4
        dim_feedforward = 64
        nlayers = 4

        self.ooi_coord_net = nn.Linear(2, embed_dim)
        self.uav_coord_net = nn.Linear(4, embed_dim)
        self.tr_encoding = nn.TransformerEncoderLayer(embed_dim, nhead,
                                                      dim_feedforward=dim_feedforward, dropout=0,
                                                      batch_first=True)
        self.tr = nn.TransformerEncoder(encoder_layer=self.tr_encoding, num_layers=nlayers)

        self.uav_encod = nn.Parameter(torch.empty(1, 3, embed_dim))
        nn.init.uniform_(self.uav_encod, -0.5, 0.5)
        self.register_buffer("uav_pad", torch.zeros((1, 3)))

    def forward(self, observations: TensorDict) -> torch.Tensor:
        ooi = observations["ooi_coords"]
        ooi_mask = 1 - observations["ooi_mask"]
        uav = observations["uav"]

        ooi = F.relu(self.ooi_coord_net(ooi))
        uav = F.relu(self.uav_coord_net(uav)) + self.uav_encod

        tr_inp = torch.cat([ooi, uav], dim=1)
        uav_pad = torch.tile(self.uav_pad, (ooi_mask.shape[0], 1))
        pad_mask = torch.cat([ooi_mask, uav_pad], dim=1)
        tr_out = self.tr(tr_inp, src_key_padding_mask=pad_mask)

        uav = torch.flatten(tr_out[:, -2:], start_dim=1)
        return uav


env = Drones()

policy_kwargs = dict(
    features_extractor_class=FE
)

model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=25000)
model.save("ppo_drones")
