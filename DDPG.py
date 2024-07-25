from collections import deque
import time
import pylab as p
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.distributions import Normal
from SAC import ReplayBuffer
from Utils import orthogonal_init


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width * 2)
        self.l3 = nn.Linear(hidden_width * 2, hidden_width)
        self.mean_layer = nn.Linear(hidden_width, action_dim)

        orthogonal_init(self.l1, 1.0)
        orthogonal_init(self.l2, 1.0)
        orthogonal_init(self.l3, 1.0)
        orthogonal_init(self.mean_layer, 0.01)

    def forward(self, x, deterministic=False, with_logprob=True):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        mean = self.mean_layer(x)
        a = torch.tanh(mean)
        return a


class Critic(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Critic, self).__init__()
        # Q1
        self.l1_1 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l2_1 = nn.Linear(hidden_width, hidden_width * 2)
        self.l3_1 = nn.Linear(hidden_width * 2, hidden_width)
        self.l4_1 = nn.Linear(hidden_width, 1)

        orthogonal_init(self.l1_1, 1.0)
        orthogonal_init(self.l2_1, 1.0)
        orthogonal_init(self.l3_1, 1.0)
        orthogonal_init(self.l4_1, 1.0)

    def forward(self, s, a):
        s_a = torch.cat([s, a], 1)
        q1 = F.relu(self.l1_1(s_a))
        q1 = F.relu(self.l2_1(q1))
        q1 = F.relu(self.l3_1(q1))
        q1 = self.l4_1(q1)

        return q1


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, hidden_width, device):
        self.hidden_width = hidden_width  # The number of neurons in hidden layers of the neural network
        self.batch_size = 256  # batch size
        self.GAMMA = 0.96  # discount factor
        self.TAU = 0.005  # Softly update the target network
        self.alr = 3e-4  # learning rate
        self.clr = 1e-3  # learning rate
        self.clip_grad_norm = 3.0
        self.device = device
        self.max_action = max_action
        self.noise_std = 0.2
        self.noise_clip = 0.5 * max_action

        self.actor = Actor(state_dim, action_dim, self.hidden_width).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.critic = Critic(state_dim, action_dim, self.hidden_width).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.alr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.alr)

        self.MseLoss = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, device, False)

    def choose_action(self, s, deterministic=False):
        s = s.unsqueeze(0)
        a = self.actor(s).data.cpu().numpy().flatten()
        if deterministic:
            action = a * self.max_action
        else:
            noise = np.clip(np.random.normal(0, self.noise_std, size=a.shape), -self.noise_clip, self.noise_clip)
            a = (a + noise).clip(-1.0, 1.0)
            action = a * self.max_action
        return action, a

    def update(self):
        batch_s, batch_a, batch_r, batch_s_, batch_dw = self.replay_buffer.sample(self.batch_size)  # Sample a batch

        # Compute the target Q
        with torch.no_grad():  # target_Q has no gradient
            Q_ = self.critic_target(batch_s_, self.actor_target(batch_s_))
            target_Q = batch_r + self.GAMMA * (1 - batch_dw) * Q_

        # Compute the current Q and the critic loss
        current_Q = self.critic(batch_s, batch_a)
        critic_loss = self.MseLoss(target_Q, current_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # nn.utils.clip_grad_norm_(parameters=self.critic.parameters(), max_norm=self.clip_grad_norm)
        self.critic_optimizer.step()

        # Freeze critic networks so you don't waste computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute the actor loss
        actor_loss = -self.critic(batch_s, self.actor(batch_s)).mean()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # nn.utils.clip_grad_norm_(parameters=self.actor.parameters(), max_norm=self.clip_grad_norm)
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        # Softly update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

        return actor_loss.detach().cpu().numpy(), critic_loss.detach().cpu().numpy()

    def load(self, a_pt, at_pt, c_pt, ct_pt):
        self.actor.load_state_dict(torch.load(a_pt))
        self.actor_target.load_state_dict(torch.load(at_pt))
        self.critic.load_state_dict(torch.load(c_pt))
        self.critic_target.load_state_dict(torch.load(ct_pt))

    def train_prep(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()

    def eval_prep(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()
