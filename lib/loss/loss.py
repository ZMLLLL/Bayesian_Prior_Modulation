import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class CrossEntropy(nn.Module):
    def __init__(self, para_dict=None):
        super(CrossEntropy, self).__init__()

    def forward(self, output, target):
        loss = F.cross_entropy(output, target)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, para_dict=None):
        super(FocalLoss, self).__init__()
        cfg = para_dict["cfg"]
        # self.gamma = cfg.LOSS.FOCAL.GAMMA
        self.gamma = 1
        assert self.gamma >= 0

    def focal_loss(self, input_values):
        """Computes the focal loss"""
        p = torch.exp(-input_values)
        loss = (1 - p) ** self.gamma * input_values
        return loss.mean()

    def forward(self, input, target):
        return self.focal_loss(F.cross_entropy(input, target, reduction='none'))


# The LDAMLoss class is copied from the official PyTorch implementation in LDAM (https://github.com/kaidic/LDAM-DRW).
class LDAMLoss(nn.Module):
    def __init__(self, para_dict=None):
        super(LDAMLoss, self).__init__()
        s = 30
        self.num_class_list = para_dict["num_class_list"]
        self.device = para_dict["device"]

        cfg = para_dict["cfg"]
        max_m = cfg.LOSS.LDAM.MAX_MARGIN
        m_list = 1.0 / np.sqrt(np.sqrt(self.num_class_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).to(self.device)
        self.m_list = m_list
        assert s > 0
        self.s = s

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.FloatTensor)
        index_float = index_float.to(self.device)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)

        return F.cross_entropy(self.s * output, target)


