import torch
import torch.nn as nn

from net.component.basic_conv import BasicConv
from net.component.basic_res_block import BasicResBlock


class CNNBackbone(nn.Module):

    def __init__(self, channel):
        super(CNNBackbone, self).__init__()
        self.conv1_311 = BasicConv(channel, channel, 3, 1, 1)
        self.res_1 = BasicResBlock(channel)
        self.conv2_312 = BasicConv(channel, 2*channel, 3, 1, 2)
        self.res_2 = BasicResBlock(2*channel)
        self.up_2 = nn.Upsample(scale_factor=2)
        self.conv3_312 = BasicConv(2*channel, 4*channel, 3, 1, 2)
        self.res_3 = BasicResBlock(4*channel)
        self.up_3 = nn.Upsample(scale_factor=4)
        self.conv4_311 = BasicConv(7*channel, channel, 3, 1, 1)

    def forward(self, x):
        x = self.conv1_311(x)
        out1 = self.res_1(x)
        x = self.conv2_312(x)
        out2 = self.res_2(x)
        out2 = self.up_2(out2)
        x = self.conv3_312(x)
        out3 = self.res_3(x)
        out3 = self.up_3(out3)
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.conv4_311(out)
        return out

