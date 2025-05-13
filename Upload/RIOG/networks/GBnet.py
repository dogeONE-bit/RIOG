import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class GBnet(nn.Module):
    def __init__(self, sync_bn = True):
        super(GBnet, self).__init__()

        if sync_bn:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.conv = nn.Sequential(nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1, bias = False),
                      BatchNorm(16),
                      nn.ReLU(),
                      nn.Dropout(0.1),
                      nn.Conv2d(16, 16, kernel_size=1, stride=1, bias = False),
                      BatchNorm(16),
                      nn.ReLU(),
                      nn.Dropout(0.1),
                      nn.Conv2d(16, 2, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x):
        x = self.conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class sGBnet(nn.Module):
    def __init__(self, sync_bn = True):
        super(sGBnet, self).__init__()

        if sync_bn:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.conv = nn.Sequential(nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias = False),
                      BatchNorm(8),
                      nn.ReLU(),
                      nn.Dropout(0.1),
                      nn.Conv2d(8, 8, kernel_size=1, stride=1, bias = False),
                      BatchNorm(1),
                      nn.ReLU(),
                      nn.Dropout(0.1),
                      nn.Conv2d(8, 1, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x):
        x = self.conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

