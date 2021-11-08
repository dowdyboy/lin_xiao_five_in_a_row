import torch
import torch.nn as nn
import torch.nn.functional as F

from net.component.spp import SPP


class ProbHead(nn.Module):

    def __init__(self, channel):
        super(ProbHead, self).__init__()
        self.spp = SPP(channel)
        self.conv_out = nn.Conv2d(channel, 1, 1, 1, 0)

    def forward(self, x):
        x = self.spp(x)
        x = self.conv_out(x)
        x = F.sigmoid(x)
        return x
