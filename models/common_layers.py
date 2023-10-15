"""
Copyright (C) 2023 Yukara Ikemiya
"""

import torch.nn as nn
import torch.nn.functional as F


# Implementation based on
# "Identity Mappings in Deep Residual Networks"
# https://arxiv.org/abs/1603.05027


class ResConv2DBlock(nn.Module):
    def __init__(self, n_in: int, act_func: str = 'mish', bn: bool = True):
        super().__init__()
        self.act = eval(f'F.{act_func}')
        self.c1 = nn.Conv2d(n_in, n_in, 3, padding=1, bias=not bn)
        self.c2 = nn.Conv2d(n_in, n_in, 1)
        self.bn1 = nn.BatchNorm2d(n_in) if bn else nn.Sequential()
        self.bn2 = nn.BatchNorm2d(n_in) if bn else nn.Sequential()

    def forward(self, x):
        x_h = self.c1(self.act(self.bn1(x)))
        x_h = self.c2(self.act(self.bn2(x_h)))
        return x + x_h


class Resnet2D(nn.Module):
    def __init__(self, n_in, n_depth, act_func='mish', bn=False):
        super().__init__()
        blocks = [ResConv2DBlock(n_in, act_func=act_func, bn=bn) for _ in range(n_depth)]
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
