import torch
import torch.nn as nn


class AlphaLoss(nn.Module):
    
    def __init__(self):
        super(AlphaLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.cel = nn.CrossEntropyLoss()

    def forward(self, p, v, gt_p, gt_v):
        loss_v = self.mse(v, gt_v)
        _, p_pred = torch.max(gt_p.view(gt_p.size(0), -1), dim=1)
        loss_p = self.cel(p.view(p.size(0), -1), p_pred)
        return loss_v + loss_p


