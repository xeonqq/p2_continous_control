import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from device import device



def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu):
        super(FCBody, self).__init__()
        dims = (state_dim,) + hidden_units
        self.layers = nn.ModuleList(
            [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x


class ActorNet(nn.Module):
    def __init__(self,
                 action_dim,
                 phi_body,
                 ):
        super(ActorNet, self).__init__()
        self.phi_body = phi_body
        self.fc_action = layer_init(nn.Linear(phi_body.feature_dim, action_dim), 1e-3)

        self.actor_params = list(self.fc_action.parameters())
        self.phi_params = list(self.phi_body.parameters())

        self.to(device)

    def forward(self, obs):
        phi = self.phi_body(obs)
        action = self.fc_action(phi)
        # to do add OU noise to action
        return torch.tanh(action)


class CriticNet(nn.Module):
    def __init__(self,
                 action_dim,
                 phi_body,
                 ):
        super(CriticNet, self).__init__()
        self.phi_body = phi_body
        self.fc_critic = layer_init(nn.Linear(phi_body.feature_dim + action_dim, 1), 1e-3)

        self.critic_params = list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())

        self.to(device)

    def forward(self, obs, action):
        phi = self.phi_body(obs)
        xs = torch.cat((phi, action), dim=1)
        value = self.fc_critic(xs)
        return value
