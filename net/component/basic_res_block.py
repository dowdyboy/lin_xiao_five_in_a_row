import torch
import torch.nn as nn
from net.component.basic_conv import BasicConv


class BasicResBlock(nn.Module):

    def __init__(self, channel):
        super(BasicResBlock, self).__init__()
        self.conv311 = BasicConv(channel, channel, 3, 1, 1)
        self.conv101 = BasicConv(channel, channel, 1, 0, 1)

    def forward(self, x):
        out = self.conv311(x)
        out = self.conv101(out)
        out = out + x
        return out
