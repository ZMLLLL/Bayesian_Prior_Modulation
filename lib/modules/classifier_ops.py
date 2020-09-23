import torch
import torch.nn as nn
import torch.nn.functional as F
import math



# for LDAM Loss

class FCNorm(nn.Module):

    def __init__(self, num_features, num_classes):
        super(FCNorm, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)



    def forward(self, x):
        # out = 20.0 * F.linear(F.normalize(x), F.normalize(self.weight))
        out = F.linear(F.normalize(x), F.normalize(self.weight))
        return out


class DistFC(nn.Module):

    def __init__(self, num_features, num_classes,init_weight=True):
        super(DistFC, self).__init__()
        self.centers=nn.Parameter(torch.randn(num_features,num_classes).cuda(),requires_grad=True)
        if init_weight:
            self.__init_weight()


    def __init_weight(self):
        nn.init.kaiming_normal_(self.centers)


    def forward(self, x):

        features_square=torch.sum(torch.pow(x,2),1, keepdim=True)
        centers_square=torch.sum(torch.pow(self.centers,2),0, keepdim=True)
        features_into_centers=2.0*torch.matmul(x, (self.centers))
        dist=features_square+centers_square-features_into_centers   
        return self.centers, dist