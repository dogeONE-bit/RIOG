import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class GBnet2(nn.Module):
    def __init__(self, sync_bn = True):
        super(GBnet2, self).__init__()

        if sync_bn:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.conv = nn.Sequential(nn.Conv2d(305, 64, kernel_size=3, stride=1, padding=1, bias = False),
                      BatchNorm(64),
                      nn.ReLU(),
                      nn.Dropout(0.1),
                      nn.Conv2d(64, 16, kernel_size=1, stride=1, bias = False),
                      BatchNorm(16),
                      nn.ReLU(),
                      nn.Dropout(0.1),
                      nn.Conv2d(16, 3, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x):
        x = self.conv(x)

        x = F.interpolate(x, size=torch.Size([512, 512]), mode='bilinear', align_corners=True)

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



