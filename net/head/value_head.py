import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueHead(nn.Module):
    
    def __init__(self, channel):
        super(ValueHead, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv_out = nn.Conv2d(channel, 1, 1, 1, 0)

    def forward(self, x):
        x = self.gap(x)
        x = self.conv_out(x)
        x = F.tanh(x)
        return x

