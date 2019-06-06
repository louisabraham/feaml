import torch
from torch import nn


def MLP(*shapes):
    layers = []
    for i, j in zip(shapes[:-1], shapes[1:]):
        m = nn.Linear(i, j)
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.zero_()
        layers.append(m)
        layers.append(nn.ReLU())
    layers.pop()
    return nn.Sequential(*layers)

