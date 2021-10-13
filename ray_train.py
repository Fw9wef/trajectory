import ray
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from env import Drones
import torch
from torch import nn
import torch.nn.functional as F


class RayDrones(Drones):
    def __init__(self, env_cofig):
        Drones.__init__(**env_cofig)


class Model(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, *args, **kwargs):
        TorchModelV2.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)
        embed_dim = model_config["embed_dim"]
        nhead = model_config["nhead"]
        dim_feedforward = model_config["dim_feedforward"]
        nlayers = model_config["nlayers"]

        self.ooi_coord_net = nn.Linear(2, embed_dim)
        self.uav_coord_net = nn.Linear(4, embed_dim)
        self.tr_encoding = nn.TransformerEncoderLayer(embed_dim, nhead,
                                                      dim_feedforward=dim_feedforward, dropout=0,
                                                      batch_first=True)
        self.tr = nn.TransformerEncoder(encoder_layer=self.tr_encoding, num_layers=nlayers)
        self.predict_head = nn.Linear(embed_dim*2, 4)
        self.value_head = nn.Linear(embed_dim*2, 1)

        self.uav_encod = nn.Parameter(torch.empty(1, 3, 16))
        nn.init.uniform_(self.uav_encode, -0.5, 0.5)
        #self.register_buffer("value", torch.zeros((1,)))
        self.register_buffer("uav_pad", torch.zeros((1, 3)))

    def forward(self, input_dict, state, seq_lens):
        ooi = input_dict["obs"]["ooi_coords"]
        ooi_mask = 1 - input_dict["obs"]["ooi_mask"]
        uav = input_dict["obs"]["uav"]

        ooi = F.relu(self.ooi_coord_net(ooi))
        uav = F.relu(self.uav_coord_net(uav)) + self.uav_encod

        tr_inp = torch.cat([ooi, uav], dim=1)
        uav_pad = torch.tile(self.uav_pad, (ooi_mask.shape[0], 1))
        pad_mask = torch.cat([ooi_mask, uav_pad], dim=1)
        tr_out = self.tr(tr_inp, src_key_padding_mask=pad_mask)

        uav = torch.flatten(tr_out[:, -2:], start_dim=1)
        out = self.predict_head(uav)
        self.value = self.value_head(uav)

        return out, None

    def value_function(self):
        return self.value


ModelCatalog.register_custom_model("my_torch_model", Model)
print("inited 0")
ray.init(include_dashboard=False)
print("inited 1")
trainer = ppo.PPOTrainer(env=RayDrones, config={
    "env_config": {},  # config to pass to env class
    "framework": "torch",
    "model": {
        "custom_model": "my_torch_model",
        # Extra kwargs to be passed to your model's c'tor.
        "custom_model_config": {"embed_dim": 16,
                                "nhead": 4,
                                "dim_feedforward": 64,
                                "nlayers": 4},
    },
})

print("inited 2")

while True:
    print(trainer.train())
