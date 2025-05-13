import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class Model_rec(nn.Module):
    def __init__(self, num_classes=2, backbone='mobilenet', BatchNorm=SynchronizedBatchNorm2d):
        super(Model_rec, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()

        self.last_conv_boundary = nn.Sequential(
                                       BatchNorm(305),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(305, 128, kernel_size=1, stride=1),
                                       BatchNorm(128),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(128, 3, kernel_size=1, stride=1))
        
        self._init_weight()


    def forward(self, x):

        x1 = self.last_conv_boundary(x)


        x1 = F.interpolate(x1, size=torch.Size([512, 512]), mode='bilinear', align_corners=True)

        return x1

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

