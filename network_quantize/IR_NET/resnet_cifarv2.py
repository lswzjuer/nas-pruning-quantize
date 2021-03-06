# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-09-19 18:06:07
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-10-14 01:45:37

import torch 
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torch.nn.functional as F
import torch.nn.init as init
from torch.hub import load_state_dict_from_url
import math
import ir_1w1a
import ir_1w32a



__all__ = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202',
            'resnet20_1w1a','resnet20_1w32a']


model_urls = {
    'resnet20': 'https://raw.githubusercontent.com/akamaster/pytorch_resnet_cifar10/master/pretrained_models/resnet20-12fca82f.th',
    'resnet32': 'https://raw.githubusercontent.com/akamaster/pytorch_resnet_cifar10/master/pretrained_models/resnet32-d509ac18.th',
    'resnet44': 'https://raw.githubusercontent.com/akamaster/pytorch_resnet_cifar10/master/pretrained_models/resnet44-014dd654.th',
    'resnet56': 'https://raw.githubusercontent.com/akamaster/pytorch_resnet_cifar10/master/pretrained_models/resnet56-4bfd9763.th',
    'resnet110': 'https://raw.githubusercontent.com/akamaster/pytorch_resnet_cifar10/master/pretrained_models/resnet110-1d1ed7c2.th.th',
    'resnet1202': 'https://raw.githubusercontent.com/akamaster/pytorch_resnet_cifar10/master/pretrained_models/resnet1202-f3b1deed.th',
}


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class BasicBlock_1w32a(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock_1w32a, self).__init__()
        self.conv1 = ir_1w32a.IRConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = ir_1w32a.IRConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     ir_1w32a.IRConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class BasicBlock_1w1a(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock_1w1a, self).__init__()
        self.conv1 = ir_1w1a.IRConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = ir_1w1a.IRConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     ir_1w1a.IRConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )


    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out += self.shortcut(x)
        out = F.hardtanh(out)
        x1 = out
        out = self.bn2(self.conv2(out))
        out += x1
        out = F.hardtanh(out)
        return out



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.block=block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.glopool= nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.linear = nn.Linear(64, num_classes)
        self.bn2 = nn.BatchNorm1d(64)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if isinstance(self.block,BasicBlock_1w1a):
            out = F.hardtanh(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.glopool(out)
        out = out.view(out.size(0), -1)
        if isinstance(self.block,BasicBlock_1w1a):
            out=self.bn2(out)
        # else:
        #     out=F.relu(self.bn2(out)) 
        out = self.linear(out)
        return out


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        s = load_state_dict_from_url(model_urls[arch], progress=progress)
        state_dict = OrderedDict()
        for k, v in s['state_dict'].items():
            if k.startswith('module.'):
                state_dict[k[7:]] = v
        model.load_state_dict(state_dict)
    return model



def resnet20_1w1a(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet20', BasicBlock_1w1a, [3, 3, 3], pretrained, progress)


def resnet20_1w32a(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet20', BasicBlock_1w32a, [3, 3, 3], pretrained, progress)


def resnet20(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet20', BasicBlock, [3, 3, 3], pretrained, progress)


def resnet32(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet32', BasicBlock, [5, 5, 5], pretrained, progress)


def resnet44(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet44', BasicBlock, [7, 7, 7], pretrained, progress)


def resnet56(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet56', BasicBlock, [9, 9, 9], pretrained, progress)


def resnet110(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet110', BasicBlock, [18, 18, 18], pretrained, progress)


def resnet1202(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet1202', BasicBlock, [200, 200, 200], pretrained, progress)




def test():
    net = resnet20(num_classes=10)
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


if __name__ == '__main__':
    test()
