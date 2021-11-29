import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaLoss(nn.Module):
    
    def __init__(self):
        super(AlphaLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.cel = nn.CrossEntropyLoss()

    def forward(self, p, v, gt_p, gt_v):
        # loss_v = self.mse(v, gt_v)
        # print('loss_v: ', v.size(), gt_v.size(), loss_v.size())
        # _, p_pred = torch.max(gt_p.view(gt_p.size(0), -1), dim=1)
        # loss_p = self.cel(p.view(p.size(0), -1), p_pred)
        # print('loss_p: ', p.size(), gt_p.size(), loss_p.size())

        v = v.view(v.size(0), -1)
        gt_v = gt_v.view(gt_v.size(0), -1)
        loss_v = F.mse_loss(v, gt_v)
        # print('loss_v: ', v.size(), gt_v.size(), loss_v.size())
        p = p.view(p.size(0), -1)
        gt_p = gt_p.view(gt_p.size(0), -1)
        loss_p = -torch.mean(torch.sum(p * gt_p, dim=-1))
        # print('loss_p: ', p.size(), gt_p.size(), loss_p.size())

        return loss_v + loss_p


