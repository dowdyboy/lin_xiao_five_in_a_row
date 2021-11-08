import torch
import torch.nn as nn

from net.backbone.cnn_backbone import CNNBackbone
from net.head.prob_head import ProbHead
from net.head.value_head import ValueHead


class LinXiaoNet(nn.Module):

    def __init__(self, channel):
        super(LinXiaoNet, self).__init__()
        self.cnn_backbone = CNNBackbone(channel)
        self.prob_head = ProbHead(channel)
        self.value_head = ValueHead(channel)

    def forward(self, x):
        x = self.cnn_backbone(x)
        prob_out = self.prob_head(x)
        value_out = self.value_head(x)
        return prob_out, value_out
