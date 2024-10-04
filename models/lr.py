"""Logistic Regression"""

import torch
import torch.nn as nn

class LR(nn.Module):
    def __init__(self, in_shape=None, out_dim=10):
        if in_shape is None:
            # assume CIFAR-10
            in_dim = 3072
        elif type(in_shape) == int:
            # D
            in_dim = in_shape
        elif len(in_shape) == 2:
            # B x D
            in_dim = in_shape[1]
        else:
            # B x C x H x W
            in_dim = in_shape[1] * in_shape[2] * in_shape[3]

        super(LR, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        out = self.linear(torch.flatten(x, start_dim=1))
        return out