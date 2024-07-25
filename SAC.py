from collections import deque
import time
import pylab as p
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.distributions import Normal
from Utils import orthogonal_init


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width, max_action, masked=False):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width * 2)
        self.l3 = nn.Linear(hidden_width * 2, hidden_width)
        self.mean_layer = nn.Linear(hidden_width, action_dim)
        self.log_std_layer = nn.Linear(hidden_width, action_dim)
        self.masked = masked
        self.mask = torch.ones((1, action_dim), requires_grad=False)
        self.mask[:, 4] = 0.
        # self.max_std = 10

        orthogonal_init(self.l1, 1.0)
        orthogonal_init(self.l2, 1.0)
        orthogonal_init(self.l3, 1.0)
        orthogonal_init(self.mean_layer, 0.01)

    def forward(self, x, deterministic=False, with_logprob=True):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)  # We output the log_std to ensure that std=exp(log_std)>0
        log_std = torch.clamp(log_std, -16, 2)
        std = torch.exp(log_std)

        dist = Normal(mean, std)  # Generate a Gaussian distribution
        if deterministic:  # When evaluating，we use the deterministic policy
            a = mean
        else:
            a = dist.rsample()  # reparameterization trick: mean+std*N(0,1)

        a_tanh = torch.tanh(a)

        if with_logprob:
            log_pi = dist.log_prob(a) - torch.log(1 - a_tanh.pow(2) + 1e-6)
            log_pi = log_pi.sum(dim=1, keepdim=True)
        else:
            log_pi = None

        # Use tanh to compress the unbounded Gaussian distribution into a bounded action interval.

        a_act = self.max_action * a_tanh

        return a_tanh, a_act, log_pi


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


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, use_suc_pool):
        self.max_size = int(1e6)
        self.count = 0
        self.size = 0
        self.max_suc_traj_num = 100
        self.device = device
        self.use_suc_pool = use_suc_pool
        self.s = np.zeros((self.max_size, state_dim))
        self.a = np.zeros((self.max_size, action_dim))
        self.r = np.zeros((self.max_size, 1))
        self.s_ = np.zeros((self.max_size, state_dim))
        self.dw = np.zeros((self.max_size, 1))

        if use_suc_pool:
            # 单条轨迹
            self.suc_pool = []
            self.suc_pool_len = 0
            self.traj_s = np.zeros((50, state_dim))
            self.traj_s_ = np.zeros((50, state_dim))
            self.traj_a = np.zeros((50, action_dim))
            self.traj_r = np.zeros((50, 1))
            self.traj_dw = np.zeros((self.max_size, 1))
            self.traj_len = 0

        self.add_size = 0

    def store(self, s, a, r, s_, dw, t_id):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.count = (self.count + 1) % self.max_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.size = min(self.size + 1, self.max_size)  # Record the number of  transitions
        self.add_size += 1

        if self.use_suc_pool:
            # 单条轨迹
            self.traj_s[self.traj_len] = s
            self.traj_a[self.traj_len] = a
            self.traj_r[self.traj_len] = r
            self.traj_s_[self.traj_len] = s_
            self.traj_dw[self.traj_len] = dw
            self.traj_len += 1

        self.add_size = 0

    def add_cur_traj_to_suc_pool(self):
        if self.use_suc_pool:
            if len(self.suc_pool) >= self.max_suc_traj_num:
                self.suc_pool_len -= self.suc_pool[0][0].shape[0]
                self.suc_pool.pop(0)
            self.suc_pool.append([self.traj_s[:self.traj_len], self.traj_a[:self.traj_len], self.traj_r[:self.traj_len],
                                  self.traj_s_[:self.traj_len], self.traj_dw[:self.traj_len]])
            self.suc_pool_len += self.traj_len
            self.traj_len = 0

    def clear_cur_traj(self):
        self.traj_len = 0

    def sample(self, batch_size):
        if self.use_suc_pool and self.suc_pool_len >= batch_size:
            index1 = np.random.choice(self.suc_pool_len, size=batch_size // 2)
            suc_s = np.concatenate([self.suc_pool[i][0] for i in range(len(self.suc_pool))], axis=0)
            suc_a = np.concatenate([self.suc_pool[i][1] for i in range(len(self.suc_pool))], axis=0)
            suc_r = np.concatenate([self.suc_pool[i][2] for i in range(len(self.suc_pool))], axis=0)
            suc_s_ = np.concatenate([self.suc_pool[i][3] for i in range(len(self.suc_pool))], axis=0)
            suc_dw = np.concatenate([self.suc_pool[i][4] for i in range(len(self.suc_pool))], axis=0)

            batch_s1 = torch.tensor(suc_s[index1], dtype=torch.float).to(self.device)
            batch_a1 = torch.tensor(suc_a[index1], dtype=torch.float).to(self.device)
            batch_r1 = torch.tensor(suc_r[index1], dtype=torch.float).to(self.device)
            batch_s_1 = torch.tensor(suc_s_[index1], dtype=torch.float).to(self.device)
            batch_dw1 = torch.tensor(suc_dw[index1], dtype=torch.float).to(self.device)

            index2 = np.random.choice(self.size, size=batch_size // 2)  # Randomly sampling
            batch_s2 = torch.tensor(self.s[index2], dtype=torch.float).to(self.device)
            batch_a2 = torch.tensor(self.a[index2], dtype=torch.float).to(self.device)
            batch_r2 = torch.tensor(self.r[index2], dtype=torch.float).to(self.device)
            batch_s_2 = torch.tensor(self.s_[index2], dtype=torch.float).to(self.device)
            batch_dw2 = torch.tensor(self.dw[index2], dtype=torch.float).to(self.device)

            batch_s = torch.cat((batch_s1, batch_s2))
            batch_a = torch.cat((batch_a1, batch_a2))
            batch_r = torch.cat((batch_r1, batch_r2))
            batch_s_ = torch.cat((batch_s_1, batch_s_2))
            batch_dw = torch.cat((batch_dw1, batch_dw2))

        else:
            index = np.random.choice(self.size, size=batch_size)  # Randomly sampling
            batch_s = torch.tensor(self.s[index], dtype=torch.float).to(self.device)
            batch_a = torch.tensor(self.a[index], dtype=torch.float).to(self.device)
            batch_r = torch.tensor(self.r[index], dtype=torch.float).to(self.device)
            batch_s_ = torch.tensor(self.s_[index], dtype=torch.float).to(self.device)
            batch_dw = torch.tensor(self.dw[index], dtype=torch.float).to(self.device)

        return batch_s, batch_a, batch_r, batch_s_, batch_dw


class PrioritizedReplayTensor(object):
    """
    Proportional Prioritization
    """
    def __init__(self, capacity, device, alpha=0.6, beta_start=0.4, beta_frames=int(1e5)):
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1  # for beta calculation
        self.capacity = capacity
        self.pos = 0
        self.device = device
        self.s = torch.empty((self.capacity, 57), dtype=torch.float32, device=self.device, requires_grad=False)
        self.a = torch.empty((self.capacity, 6), dtype=torch.float32, device=self.device, requires_grad=False)
        self.r = torch.empty((self.capacity, 1), dtype=torch.float32, device=self.device, requires_grad=False)
        self.s_ = torch.empty((self.capacity, 57), dtype=torch.float32, device=self.device, requires_grad=False)
        self.d = torch.empty((self.capacity, 1), dtype=torch.float32, device=self.device, requires_grad=False)
        self.tid = torch.empty((self.capacity, 1), dtype=torch.int, device=self.device, requires_grad=False)
        self.priorities = torch.empty((self.capacity, 1), dtype=torch.float32, device=self.device, requires_grad=False)
        self.real_len = 0
        self.iter = 0
        self.max_prio = -torch.inf

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def store(self, state, action, reward, next_state, done, t_id):
        assert state.ndim == next_state.ndim
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        t_id = torch.tensor(t_id, dtype=torch.int, device=self.device).unsqueeze(0).unsqueeze(0)
        d = torch.tensor(int(done), dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

        if self.iter >= self.capacity:
            p = self.priorities[0]  # pop oldest transition
            if p >= self.max_prio:
                self.max_prio = self.priorities.max()

            max_prio = self.max_prio.unsqueeze(0).unsqueeze(
                0) if self.real_len > 0 and self.max_prio > 0 else 1.0  # gives max priority if buffer is not empty else 1
            self.s = torch.cat((self.s[1:], state))
            self.a = torch.cat((self.a[1:], action))
            self.r = torch.cat((self.r[1:], reward))
            self.s_ = torch.cat((self.s_[1:], next_state))
            self.d = torch.cat((self.d[1:], d))
            self.tid = torch.cat((self.tid[1:], t_id))
            self.priorities = torch.cat((self.priorities[1:], max_prio))

        else:
            max_prio = self.max_prio if self.real_len > 0 and self.max_prio > 0 else 1.0  # gives max priority if buffer is not empty else 1
            pos = self.iter
            self.s[pos] = state
            self.a[pos] = action
            self.r[pos] = reward
            self.s_[pos] = next_state
            self.d[pos] = d
            self.tid[pos] = t_id
            self.priorities[pos] = max_prio

        self.real_len = min(self.real_len + 1, self.capacity)
        self.iter += 1

    def sample(self, batch_size, c_k=None):
        N = self.real_len
        if c_k > N:
            c_k = N

        if c_k == self.capacity:
            priors = self.priorities
        else:
            priors = self.priorities[self.real_len - c_k:self.real_len]

        # (priors)
        # calc P = p^a/sum(p^a)
        probs = priors ** self.alpha
        P = (probs / probs.sum()).cpu().numpy().flatten()

        # gets the indices depending on the probability p and the c_k range of the buffer
        indices = torch.tensor(np.random.choice(c_k, batch_size, p=P)).detach()

        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        # Compute importance-sampling weight
        weights = (c_k * P[indices]) ** (-beta)
        # normalize weights
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float).to(self.device).unsqueeze(-1).detach()

        batch_s = self.s[indices].clone().detach()
        batch_a = self.a[indices].clone().detach()
        batch_r = self.r[indices].clone().detach()
        batch_s_ = self.s_[indices].clone().detach()
        batch_dw = self.d[indices].clone().detach()
        batch_tid = self.tid[indices].clone().detach()


        return batch_s, batch_a, batch_r, batch_s_, batch_dw, batch_tid, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        self.priorities[batch_indices] = batch_priorities
        max_b_p = batch_priorities.max()
        if batch_priorities.max() > self.max_prio:
            self.max_prio = max_b_p

    def __len__(self):
        return self.real_len


class SACAgent(object):
    def __init__(self, state_dim, action_dim, hidden_width, max_action, device, use_ERE=False, use_suc_pool=False, mask=False):
        self.device = device
        self.max_action = max_action
        self.hidden_width = hidden_width  # The number of neurons in hidden layers of the neural network
        self.batch_size = 256  # batch size
        self.GAMMA = 0.96  # discount factor
        self.TAU = 5e-3  # Softly update the target network
        self.lr = 3e-4  # learning rate
        self.repeat_times = 1.0
        self.clip_grad_norm = 3.0
        self.use_ERE = use_ERE
        self.adaptive_alpha = True  # Whether to automatically learn the temperature alpha
        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = -action_dim
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            self.log_alpha = torch.tensor((-1,), dtype=torch.float32, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().to(self.device)
            self.alpha_optimizer = torch.optim.AdamW([self.log_alpha], lr=self.lr)
        else:
            self.alpha = 0.2

        self.actor = Actor(state_dim, action_dim, self.hidden_width, max_action, mask).to(self.device)
        self.critic = Critic(state_dim, action_dim, self.hidden_width).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        if self.use_ERE:
            self.replay_buffer = PrioritizedReplayTensor(capacity=int(1e6), device=device)
        else:
            self.replay_buffer = ReplayBuffer(state_dim, action_dim, device, use_suc_pool)

        self.update_cnt = 0
        self.difficult_coe = torch.tensor([0.0820345, 0.28467329, 0.0272534, 0.0272534, 0.18823858, 0.12145276,
                                           0.12145276, 0.14764131], dtype=torch.float32, device=self.device)

    def load(self, a_pt, c_pt):
        self.actor.load_state_dict(torch.load(a_pt))
        self.critic.load_state_dict(torch.load(c_pt))
        self.critic_target.load_state_dict(torch.load(c_pt))

    def train_prep(self):
        self.actor.train()
        self.critic.train()
        self.critic_target.train()

    def eval_prep(self):
        self.actor.eval()
        self.critic.eval()
        self.critic_target.eval()

    def choose_action(self, s, deterministic=False):
        s = torch.unsqueeze(s, 0)
        # When choosing actions, we do not need to compute log_pi
        a_tanh, a_act, _ = self.actor(s, deterministic, False)
        return a_tanh.data.cpu().numpy().flatten(), a_act.data.cpu().numpy().flatten()

    def update(self, c_k=None):
        # update_times = int(self.repeat_times * self.replay_buffer.add_size)
        update_times = 1
        self.replay_buffer.add_size = 0
        a_losses = []
        c_losses = []
        alpha_losses = []
        assert update_times >= 1
        for _ in range(update_times):

            if self.use_ERE:
                batch_s, batch_a, batch_r, batch_s_, batch_dw, batch_tid, indices, weights = self.replay_buffer.sample(
                    self.batch_size, c_k)
            else:
                batch_s, batch_a, batch_r, batch_s_, batch_dw = self.replay_buffer.sample(
                    self.batch_size)  # Sample a batch
                indices = []
                weights = 1.0

            with torch.no_grad():
                batch_a_, _, log_pi_ = self.actor(batch_s_)  # a_ from the current policy
                # Compute target Q
                target_Q1, target_Q2 = self.critic_target(batch_s_, batch_a_)
                target_Q = batch_r + self.GAMMA * (1 - batch_dw) * (
                            torch.min(target_Q1, target_Q2) - self.alpha * log_pi_)

            # Compute current Q
            current_Q1, current_Q2 = self.critic(batch_s, batch_a)
            td_error1 = target_Q.detach() - current_Q1
            td_error2 = target_Q.detach() - current_Q2
            # Compute critic loss
            if self.use_ERE:
                critic1_loss = 0.5 * (td_error1.pow(2) * weights).mean()
                critic2_loss = 0.5 * (td_error2.pow(2) * weights).mean()
                critic_loss = critic1_loss + critic2_loss
            else:
                critic_loss = (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(parameters=self.critic.parameters(), max_norm=self.clip_grad_norm)
            self.critic_optimizer.step()

            a, _, log_pi = self.actor(batch_s)
            # Update alpha
            if self.adaptive_alpha:
                alpha_loss = -(self.log_alpha.exp().to(self.device) * (log_pi + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                nn.utils.clip_grad_norm_(parameters=self.alpha_optimizer.param_groups[0]["params"],
                                         max_norm=self.clip_grad_norm)
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp().to(self.device)

            # Softly update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

            # Freeze critic networks so you don't waste computational effort
            for params in self.critic.parameters():
                params.requires_grad = False

            with torch.no_grad():
                self.log_alpha[:] = self.log_alpha.clamp(-16, 2)

            # Compute actor loss
            Q1, Q2 = self.critic(batch_s, a)
            Q = torch.min(Q1, Q2)
            td_error3 = (self.alpha * log_pi - Q)
            actor_loss = ((self.alpha * log_pi - Q) * weights).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(parameters=self.actor.parameters(), max_norm=self.clip_grad_norm)
            self.actor_optimizer.step()

            # update priorities
            if self.use_ERE:
                with torch.no_grad():

                    # difficult_coe v2
                    priors = (torch.abs(td_error1) * 0.5 + torch.abs(td_error2) * 0.5 + torch.abs(
                        td_error3) * 0.1 + 1e-5) * self.difficult_coe[batch_tid]

                    self.replay_buffer.update_priorities(indices, priors)

            # Freeze critic networks so you don't waste computational effort
            for params in self.critic.parameters():
                params.requires_grad = True

            a_losses.append(actor_loss.item())
            c_losses.append(critic_loss.item())
            alpha_losses.append(alpha_loss.item())

        return np.mean(a_losses), np.mean(c_losses), np.mean(alpha_losses)

