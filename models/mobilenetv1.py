# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-09-16 17:14:39
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-11-22 19:28:36
'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    # NOTE: change (128,2) stride 2 -> 1 for CIFAR10 (128,1)
    # imagenet downsample 32
    # cifar10 downsample 8
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.glopool= nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.glopool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def MobileNetV1(pretrained=False, progress=True, **kwargs):
    return MobileNet(**kwargs)


def test():
    net = MobileNetV1(num_classes=10)
    x = torch.randn(1,3,64,64)
    y = net(x)
    print(y.size())

if __name__ == '__main__':
    test()
