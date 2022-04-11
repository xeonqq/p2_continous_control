import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from device import device
from model import ActorNet, CriticNet, FCBody
from ou_noise import OUNoise
from replay_buffer import ReplayBuffer

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 1e-3  # learning rate of the critic
WEIGHT_DECAY = 1e-2  # L2 weight decay


class DDPG_Agent(object):
    def __init__(self, state_dim, action_dim, seed=42):
        print("Using device: ", device)
        self._actor_local = ActorNet(action_dim, FCBody(state_dim, (400, 300)))
        self._actor_target = ActorNet(action_dim, FCBody(state_dim, (400, 300)))
        self._actor_optimizer = optim.Adam(self._actor_local.parameters(), lr=LR_ACTOR)

        self._critic_local = CriticNet(action_dim, FCBody(state_dim, (400,)), FCBody(400 + action_dim, (300,)))
        self._critic_target = CriticNet(action_dim, FCBody(state_dim, (400,)), FCBody(400 + action_dim, (300,)))
        self._critic_optimizer = optim.Adam(self._critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        self._replay_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        self._ou_noise = OUNoise(action_dim, seed)
        self._noise_decay = 0.99
        
        self.soft_update(self._actor_local, self._actor_target, 1)
        self.soft_update(self._critic_local, self._critic_target, 1)

    def load_actor_model(self, model):
        self._actor_target.load_state_dict(torch.load(model))

    def step(self, state, action, reward, next_state, done):
        self._replay_buffer.add(state, action, reward, next_state, done)
        # Learn, if enough samples are available in memory
        if len(self._replay_buffer) > BATCH_SIZE:
            experiences = self._replay_buffer.sample()
            self.learn(experiences)

    def reset(self):
        self._ou_noise.reset()

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        next_actions = self._actor_target(next_states)
        # target_return = rewards + GAMMA * self._critic_target(next_states, next_actions).detach()
        target_next_Q = self._critic_target(next_states, next_actions)
        target_return = rewards + GAMMA * target_next_Q * (1 - dones)
        expected_return = self._critic_local(states, actions)

        critic_loss = F.mse_loss(target_return, expected_return)
        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_optimizer.step()

        expected_action = self._actor_local(states)
        # self._critic_local.eval()
        # with torch.no_grad():
        actor_loss = -self._critic_local(states, expected_action).mean()
        # self._critic_local.train()

        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        self._actor_optimizer.step()

        self.soft_update(self._actor_local, self._actor_target, TAU)
        self.soft_update(self._critic_local, self._critic_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def act(self, state, noise):
        state = torch.from_numpy(state).float().to(device)
        self._actor_local.eval()
        with torch.no_grad():
            action = self._actor_local(state)
        self._actor_local.train()
        action = action.cpu().data.numpy()
        if noise:
            noise = self._ou_noise.sample() * self._noise_decay
            self._noise_decay *= self._noise_decay
            action += noise
        return np.clip(action, -1, 1)  # all actions between -1 and 1
