# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-10-02 21:04:48
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-10-03 20:13:58

import torch
import math
from torch import nn
import torch.nn.functional as F
import layer as Layer



class baseBlock(nn.Module):
    """docstring for """
    def __init__(self, in_planes, planes, kernel_size,stride,padding,binarynum=1,
                        pooling=False,act=None,dilation=1,groups=1,bias=False):
        super(baseBlock, self).__init__()

        self.conv = nn.Conv2d(in_planes,planes,kernel_size=kernel_size,stride=stride,padding=padding,
            dilation=dilation,groups=groups,bias=bias)
        self.bn = nn.BatchNorm2d(planes)
        if act == "relu":
            self.act  = nn.ReLU(inplace=True)
        elif act == "htanh":
            self.act = nn.Hardtanh(inplace=True)
        else:
            self.act = None
        if pooling:
            self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pooling = None

    def forward(self,input):
        out = self.conv(input)
        if self.pooling:
            out = self.pooling(out)
        out = self.bn(out)
        if self.act:
            out = self.act(out)
        return out



class baseBlock_1w1a_res(nn.Module):
    """docstring for """
    def __init__(self, in_planes, planes, kernel_size,stride,padding,binarynum=1,
                        pooling=False,act=None,dilation=1,groups=1,bias=False):
        super(baseBlock_1w1a_res, self).__init__()

        self.conv = Layer.BNNConv2d_1w1a(in_planes,planes,kernel_size=kernel_size,stride=stride,padding=padding,
            dilation=dilation,groups=groups,bias=bias)
        self.bn = nn.BatchNorm2d(planes)
        if act == "relu":
            self.act  = nn.ReLU(inplace=True)
        elif act == "htanh":
            self.act = nn.Hardtanh(inplace=True)
        else:
            self.act = None

        if pooling:
            self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pooling = None

        self.shortcut=nn.Sequential()
        if pooling:
            self.shortcut=nn.Sequential(
                            nn.MaxPool2d(kernel_size=2, stride=2)
            )

    def forward(self,input):
        out = self.conv(input)
        if self.pooling:
            out = self.pooling(out)
        out = self.bn(out)
        out = out + self.shortcut(input)
        if self.act:
            out = self.act(out)
        return out




class Lenet(nn.Module):
    def __init__(self, num_classes=10,binarynum=1):
        super(Lenet, self).__init__()
        self.num_classes=num_classes
        self.block0 = baseBlock(in_planes=1,planes=32,kernel_size=3,stride=1,padding=1,bias=False,
                                pooling=True,act=None)
        self.block1 = baseBlock_1w1a_res(in_planes=32,planes=32,kernel_size=3,stride=1,padding=1,bias=False,
                                pooling=True,act=None)
        self.block2 = baseBlock_1w1a_res(in_planes=32,planes=32,kernel_size=3,stride=1,padding=1,bias=False,
                                pooling=True,act=None)   
        self.fc = nn.Linear(288, self.num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, Layer.BNNConv2d_1w1a):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def test():
    net = Lenet(num_classes=10)
    x = torch.randn(2,1,28,28)
    y = net(x)
    print(y.size())
    print(net)




if __name__ == '__main__':
    test()
