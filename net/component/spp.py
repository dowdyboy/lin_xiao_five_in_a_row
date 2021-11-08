import torch
import torch.nn as nn
from net.component.basic_res_block import BasicConv


class SPP(nn.Module):

    def __init__(self, channel):
        super(SPP, self).__init__()
        self.conv311 = BasicConv(channel, channel, 3, 1, 1)
        self.pool311 = nn.MaxPool2d(3, 1, 1)
        self.pool521 = nn.MaxPool2d(5, 1, 2)
        self.pool941 = nn.MaxPool2d(9, 1, 4)
        self.conv101 = BasicConv(4 * channel, channel, 1, 0, 1)

    def forward(self, x):
        x = self.conv311(x)
        p311 = self.pool311(x)
        p512 = self.pool521(x)
        p941 = self.pool941(x)
        x = torch.cat([x, p311, p512, p941], dim=1)
        x = self.conv101(x)
        return x
