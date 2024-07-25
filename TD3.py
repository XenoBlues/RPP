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
    def __init__(self, state_dim, action_dim, hidden_width, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width * 2)
        self.l3 = nn.Linear(hidden_width * 2, hidden_width)
        self.mean_layer = nn.Linear(hidden_width, action_dim)

        orthogonal_init(self.l1, 1.0)
        orthogonal_init(self.l2, 1.0)
        orthogonal_init(self.l3, 1.0)
        orthogonal_init(self.mean_layer, 0.01)

    def forward(self, x, deterministic=False, with_logprob=True):
        x = F.tanh(self.l1(x))
        x = F.tanh(self.l2(x))
        x = F.tanh(self.l3(x))
        mean = self.mean_layer(x)
        a = torch.tanh(mean) * self.max_action
        return a


class Critic(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Critic, self).__init__()
        # Q1
        self.l1_1 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l2_1 = nn.Linear(hidden_width, hidden_width * 2)
        self.l3_1 = nn.Linear(hidden_width * 2, hidden_width)
        self.l4_1 = nn.Linear(hidden_width, 1)
        # Q2
        self.l1_2 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l2_2 = nn.Linear(hidden_width, hidden_width * 2)
        self.l3_2 = nn.Linear(hidden_width * 2, hidden_width)
        self.l4_2 = nn.Linear(hidden_width, 1)

        orthogonal_init(self.l1_1, 1.0)
        orthogonal_init(self.l2_1, 1.0)
        orthogonal_init(self.l3_1, 1.0)
        orthogonal_init(self.l4_1, 1.0)
        orthogonal_init(self.l1_2, 1.0)
        orthogonal_init(self.l2_2, 1.0)
        orthogonal_init(self.l3_2, 1.0)
        orthogonal_init(self.l4_2, 1.0)

    def forward(self, s, a):
        s_a = torch.cat([s, a], 1)
        q1 = F.relu(self.l1_1(s_a))
        q1 = F.relu(self.l2_1(q1))
        q1 = F.relu(self.l3_1(q1))
        q1 = self.l4_1(q1)

        q2 = F.relu(self.l1_2(s_a))
        q2 = F.relu(self.l2_2(q2))
        q2 = F.relu(self.l3_2(q2))
        q2 = self.l4_2(q2)

        return q1, q2

    def Q1(self, s, a):
        s_a = torch.cat([s, a], 1)
        q1 = F.relu(self.l1_1(s_a))
        q1 = F.relu(self.l2_1(q1))
        q1 = F.relu(self.l3_1(q1))
        q1 = self.l4_1(q1)
        return q1


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, device):
        self.max_action = max_action
        self.device = device
        self.hidden_width = 128  # The number of neurons in hidden layers of the neural network
        self.batch_size = 256  # batch size
        self.GAMMA = 0.96  # discount factor
        self.TAU = 0.005  # Softly update the target network
        self.lr = 3e-4  # learning rate
        self.explore_noise = 0.2
        self.policy_noise = 0.2  # The noise for the trick 'target policy smoothing'
        self.noise_clip = 0.5    # Clip the noise
        self.policy_freq = 2  # The frequency of policy updates
        self.actor_pointer = 0

        self.actor = Actor(state_dim, action_dim, self.hidden_width, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.critic = Critic(state_dim, action_dim, self.hidden_width).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.replay_buffer = ReplayBuffer(state_dim, action_dim, device, False)

    def choose_action(self, s, deterministic=False):
        s = s.unsqueeze(0)
        a = self.actor(s).data.cpu().numpy().flatten()
        if deterministic:
            action = a
        else:
            noise = np.clip(np.random.randn(*a.shape) * self.explore_noise, -self.noise_clip, self.noise_clip) * self.max_action

            action = (a + noise).clip(-self.max_action, self.max_action)
        return action, a

    def update(self):
        batch_s, batch_a, batch_r, batch_s_, batch_dw = self.replay_buffer.sample(self.batch_size)  # Sample a batch

        if batch_s is not None:
            self.actor_pointer += 1
            # Compute the target Q
            with torch.no_grad():  # target_Q has no gradient
                # Trick 1:target policy smoothing
                # torch.randn_like can generate random numbers sampled from N(0,1)，which have the same size as 'batch_a'
                noise = (torch.randn_like(batch_a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip) * self.max_action
                next_action = (self.actor_target(batch_s_) + noise).clamp(-self.max_action, self.max_action)

                # Trick 2:clipped double Q-learning
                target_Q1, target_Q2 = self.critic_target(batch_s_, next_action)
                target_Q = batch_r + self.GAMMA * (1 - batch_dw) * torch.min(target_Q1, target_Q2)

            # Get the current Q
            current_Q1, current_Q2 = self.critic(batch_s, batch_a)
            # Compute the critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Trick 3:delayed policy updates
            if self.actor_pointer % self.policy_freq == 0:
                # Freeze critic networks so you don't waste computational effort
                for params in self.critic.parameters():
                    params.requires_grad = False

                # Compute actor loss
                actor_loss = -self.critic.Q1(batch_s, self.actor(batch_s)).mean()  # Only use Q1
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Unfreeze critic networks
                for params in self.critic.parameters():
                    params.requires_grad = True

                # Softly update the target networks
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)
            else:
                actor_loss = torch.tensor(0.).to(self.device)

        else:
            actor_loss = torch.tensor(0.).to(self.device)
            critic_loss = torch.tensor(0.).to(self.device)

        return actor_loss.detach().cpu().numpy(), critic_loss.detach().cpu().numpy()

    def load(self, a_pt, c_pt, at_pt, ct_pt):
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
