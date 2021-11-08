import torch
import torch.nn as nn


class BasicConv(nn.Module):

    def __init__(self, in_channel, out_channel, k, p, s):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, k, s, p)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
