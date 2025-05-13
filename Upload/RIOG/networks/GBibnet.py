import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class GBibnet(nn.Module):
    def __init__(self, sync_bn = True):
        super(GBibnet, self).__init__()

        if sync_bn:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.conv = nn.Sequential(nn.Conv2d(1, 4, kernel_size=1, stride=1),
                      BatchNorm(4),
                      nn.ReLU(),
                      nn.Dropout(0.1),
                      nn.Conv2d(4, 8, kernel_size=1, stride=1),
                      BatchNorm(8),
                      nn.ReLU(),
                      nn.Dropout(0.1))
        
        self.conv2 = nn.Sequential(nn.Conv2d(1, 4, kernel_size=1, stride=1),
                      BatchNorm(4),
                      nn.ReLU(),
                      nn.Dropout(0.1),
                      nn.Conv2d(4, 8, kernel_size=1, stride=1),
                      BatchNorm(8),
                      nn.ReLU(),
                      nn.Dropout(0.1))
        
        self.fc = nn.Linear(16, 1)
        self._init_weight()


    def forward(self, x):

        B, C, H, W = x.shape
        x1 = x[:,:1,:,:]
        x2 = x[:,1:,:,:]

        x1 = self.conv(x1).permute(0,2,3,1).reshape(B, H*W, -1)
        x2 = self.conv(x2).permute(0,2,3,1).reshape(B, H*W, -1)

        out = self.fc(torch.cat([x1, x2], dim=-1))

        out = out.reshape(B,H,W)

        return out

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
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



