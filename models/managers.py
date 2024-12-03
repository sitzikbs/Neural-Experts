# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
# This file contains the code for different manager architecture implementations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from models.pointnet import PointNetfeat
from models.modules import ParallelFullyConnectedNN
from models.modules import InputEncoder, FullyConnectedNN


class DummyModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, *args, **kwargs):
        return args[0]

class DummyManager(nn.Module):
    def __init__(self, n_experts):
        super().__init__()
        self.n_experts = n_experts
    def forward(self, points):
        return torch.zeros(points.shape[0], self.n_experts, device=points.device), (None, None)

class DummyBias(nn.Module):
    def __init__(self, n_experts):
        super().__init__()
        self.n_experts = n_experts
    def forward(self, points, q):
        return q

class ManagerConditioner(nn.Module):
    def __init__(self, manager_conditioning, laster_layer_dim=128, expert_decoder=None):
        super().__init__()
        self.manager_conditioning = manager_conditioning

    def forward(self, x, manager_input, **kwargs):
        if self.manager_conditioning == 'max':
            point_rep = torch.max(x, dim=1)[0].unsqueeze(1).expand(-1, manager_input.shape[1], -1)  # global information for the manager
        elif self.manager_conditioning == 'mean':
            point_rep = torch.mean(x, dim=1).unsqueeze(1).expand(-1, manager_input.shape[1], -1)
        elif self.manager_conditioning == 'cat':
            point_rep = x
        else:
            point_rep = None

        if point_rep is not None:
            manager_input = torch.cat([manager_input, point_rep], dim=-1)

        return manager_input
