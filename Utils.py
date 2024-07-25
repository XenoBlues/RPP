import numpy as np
import torch
from torch import nn
from typing import List, Union
import scipy


class AABB:
    def __init__(self, center, length=180, width=180, height=300):
        self.center = center
        self.length = length
        self.width = width
        self.height = height
        self.xmin = self.center[0] - self.length * 0.5
        self.xmax = self.center[0] + self.width * 0.5
        self.ymin = self.center[1] - self.width * 0.5
        self.ymax = self.center[1] + self.width * 0.5
        self.zmin = self.center[2] - self.height * 0.5
        self.zmax = self.center[2] + self.height * 0.5


class Sphere(object):
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius
        self.geo_dim = radius


class Capsule(object):
    def __init__(self, tip: Union[List[float], np.ndarray], base: Union[List[float], np.ndarray], radius: float):
        self.tip = np.array(tip)
        self.base = np.array(base)
        self.center = (self.tip + self.base) / 2.0
        self.radius = radius
        self.half_height = np.linalg.norm(self.tip - self.base) / 2.0
        self.up = (self.tip - self.base) / np.linalg.norm(self.tip - self.base)
        self.geo_dim = self.half_height

    def set(self, tip, base):
        self.tip = np.array(tip)
        self.base = np.array(base)
        self.center = (self.tip + self.base) / 2.0
        self.half_height = np.linalg.norm(self.tip - self.base) / 2.0
        self.up = (self.tip - self.base) / np.linalg.norm(self.tip - self.base)


class Cylinder:
    def __init__(self, tip, base, radius):
        self.tip = np.array(tip)
        self.base = np.array(base)
        self.center = (self.tip + self.base) / 2.0
        self.radius = radius
        self.half_height = np.linalg.norm(self.tip - self.base) / 2.0
        self.up = (self.tip - self.base) / np.linalg.norm(self.tip - self.base)
        self.geo_dim = np.sqrt(self.half_height * 0.5 + radius * radius)

    def set(self, tip, base):
        self.tip = np.array(tip)
        self.base = np.array(base)
        self.center = (self.tip + self.base) / 2.0
        self.half_height = np.linalg.norm(self.tip - self.base) / 2.0
        self.up = (self.tip - self.base) / np.linalg.norm(self.tip - self.base)


def orthogonal_init(layer, gain=1.0):
    torch.nn.init.orthogonal_(layer.weight, gain=gain)
    torch.nn.init.constant_(layer.bias, 0)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module


class FixedCategorical(torch.distributions.Categorical):
    def sample(self, sample_shape=torch.Size()):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1, keepdim=True)

    def mode(self):
        return self.mean


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(DiagGaussian, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)  # 方差
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
        return x


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)


def traj_interpolate(joint_traj, time_list, t):
    rbf_x = scipy.interpolate.CubicSpline(time_list, joint_traj[..., 0])
    rbf_y = scipy.interpolate.CubicSpline(time_list, joint_traj[..., 1])
    rbf_z = scipy.interpolate.CubicSpline(time_list, joint_traj[..., 2])
    x = rbf_x(t).reshape(-1, 1)
    y = rbf_y(t).reshape(-1, 1)
    z = rbf_z(t).reshape(-1, 1)
    return np.concatenate([x, y, z], axis=-1)


