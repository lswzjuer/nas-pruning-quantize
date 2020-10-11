# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-09-29 22:39:48
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-09-30 00:04:10


import torch 
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import bnn_layers as Layer



__all__ = ['vgg_small_cpba',
            'vgg_small_1w1alayer','vgg_small_1w1achannel']



class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class MoveBlock(nn.Module):
    """docstring for """
    def __init__(self,out_channels):
        super(MoveBlock, self).__init__()
        self.out_channels = out_channels
        self.move1 = LearnableBias(self.out_channels)
        self.act1 = nn.PReLU(self.out_channels)
        self.move2 = LearnableBias(self.out_channels)

    def forward(self,input):
        out = self.move1(input)
        out = self.act1(out)
        out = self.move2(out)
        return out


class ScaleAndShift(nn.Module):
    """docstring for ScaleAndShift"""
    def __init__(self, out_channels):
        super(ScaleAndShift, self).__init__()
        self.alphas = nn.Parameter(torch.ones(1,out_channels), requires_grad=True)
        self.betas = nn.Parameter(torch.zeros(1,out_channels), requires_grad=True)

    def forward(self, x):
        out = self.alphas.expand_as(x) * x + self.betas.expand_as(x)
        return out




class VggBlock(nn.Module):
    """docstring for """
    def __init__(self, in_planes, planes, kernel_size,stride,padding,binarynum=1,
                        pooling=False,act=None,dilation=1,groups=1,bias=False):
        super(VggBlock, self).__init__()

        self.conv = nn.Conv2d(in_planes,planes,kernel_size=kernel_size,stride=stride,padding=padding,
            dilation=dilation,groups=groups,bias=bias)
        self.bn = nn.BatchNorm2d(planes)

        if pooling:
            self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pooling = None

        if act == "relu":
            self.act  = nn.ReLU(inplace=True)
        elif act == "htanh":
            self.act = nn.Hardtanh(inplace=True)
        else:
            self.act = None

    def forward(self,input):
        out = self.conv(input)
        if self.pooling:
            out = self.pooling(out)
        out = self.bn(out)
        if self.act:
            out = self.act(out)
        return out



class VggBlock_1w1aChannel(nn.Module):
    """docstring for """
    def __init__(self, in_planes, planes, kernel_size,stride,padding,binarynum=1,
                        pooling=False,act=None,dilation=1,groups=1,bias=False):
        super(VggBlock_1w1aChannel, self).__init__()

        self.conv = Layer.BNNConv2d_1wnaChannel(in_planes,planes,binarynum=binarynum,
                                                kernel_size=kernel_size,stride=stride,padding=padding,
                                                dilation=dilation,groups=groups,bias=bias)
        self.bn = nn.BatchNorm2d(planes)
        if pooling:
            self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pooling = None
        if act == "relu":
            self.act  = nn.ReLU(inplace=True)
        elif act == "htanh":
            self.act = nn.Hardtanh(inplace=True)
        else:
            self.act = None

    def forward(self,input):
        out = self.conv(input)
        if self.pooling:
            out = self.pooling(out)
        out = self.bn(out)
        if self.act:
            out = self.act(out)
        return out
  
 

class VggBlock_1w1aLayer(nn.Module):
    """docstring for """
    def __init__(self, in_planes, planes, kernel_size,stride,padding,binarynum=1,
                        pooling=False,act=None,dilation=1,groups=1,bias=False):
        super(VggBlock_1w1aLayer, self).__init__()

        self.conv = Layer.BNNConv2d_1wnaLayer(in_planes,planes,binarynum=binarynum,
                                                kernel_size=kernel_size,stride=stride,padding=padding,
                                                dilation=dilation,groups=groups,bias=bias)
        self.bn = nn.BatchNorm2d(planes)
        if pooling:
            self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pooling = None

        if act == "relu":
            self.act  = nn.ReLU(inplace=True)
        elif act == "htanh":
            self.act = nn.Hardtanh(inplace=True)
        else:
            self.act = None

    def forward(self,input):
        out = self.conv(input)
        if self.pooling:
            out = self.pooling(out)
        out = self.bn(out)
        if self.act:
            out = self.act(out)
        return out
  




class VGG_SMALL_CPBA(nn.Module):
    def __init__(self, num_classes=10,binarynum=1):
        super(VGG_SMALL_CPBA, self).__init__()
        self.num_classes=num_classes
        self.block0 = VggBlock(in_planes=3,planes=128,kernel_size=3,stride=1,padding=1,bias=False,
                                pooling=False,act="relu")
        self.block1 = VggBlock(in_planes=128,planes=128,kernel_size=3,stride=1,padding=1,bias=False,
                                pooling=True,act="relu")
        self.block2 = VggBlock(in_planes=128,planes=256,kernel_size=3,stride=1,padding=1,bias=False,
                                pooling=False,act="relu")
        self.block3 = VggBlock(in_planes=256,planes=256,kernel_size=3,stride=1,padding=1,bias=False,
                                pooling=True,act="relu")
        self.block4 = VggBlock(in_planes=256,planes=512,kernel_size=3,stride=1,padding=1,bias=False,
                                pooling=False,act="relu")
        self.block5 = VggBlock(in_planes=512,planes=512,kernel_size=3,stride=1,padding=1,bias=False,
                                pooling=True,act="relu")
        self.fc = nn.Linear(512*4*4, self.num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
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
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x




class VGG_SMALL_1w1aLayer(nn.Module):
    def __init__(self, num_classes=10,binarynum=3):
        super(VGG_SMALL_1w1aLayer, self).__init__()
        self.num_classes=num_classes
        self.block0 = VggBlock_1w1aLayer(in_planes=3,planes=128,kernel_size=3,stride=1,padding=1,bias=False,
                                binarynum=binarynum,pooling=False,act=None)
        self.block1 = VggBlock_1w1aLayer(in_planes=128,planes=128,kernel_size=3,stride=1,padding=1,bias=False,
                                binarynum=binarynum,pooling=True,act=None)
        self.block2 = VggBlock_1w1aLayer(in_planes=128,planes=256,kernel_size=3,stride=1,padding=1,bias=False,
                                binarynum=binarynum,pooling=False,act=None)
        self.block3 = VggBlock_1w1aLayer(in_planes=256,planes=256,kernel_size=3,stride=1,padding=1,bias=False,
                                binarynum=binarynum,pooling=True,act=None)
        self.block4 = VggBlock_1w1aLayer(in_planes=256,planes=512,kernel_size=3,stride=1,padding=1,bias=False,
                               binarynum=binarynum,pooling=False,act=None)
        self.block5 = VggBlock_1w1aLayer(in_planes=512,planes=512,kernel_size=3,stride=1,padding=1,bias=False,
                                binarynum=binarynum,pooling=True,act=None)
        self.fc = nn.Linear(512*4*4, self.num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, Layer.BNNConv2d_1wnaLayer):
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
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x




class VGG_SMALL_1w1aChannel(nn.Module):
    def __init__(self, num_classes=10,binarynum=1):
        super(VGG_SMALL_1w1aChannel, self).__init__()
        self.num_classes=num_classes
        self.block0 = VggBlock_1w1aChannel(in_planes=3,planes=128,kernel_size=3,stride=1,padding=1,bias=False,
                                binarynum=binarynum,pooling=False,act=None)
        self.block1 = VggBlock_1w1aChannel(in_planes=128,planes=128,kernel_size=3,stride=1,padding=1,bias=False,
                                binarynum=binarynum,pooling=True,act=None)
        self.block2 = VggBlock_1w1aChannel(in_planes=128,planes=256,kernel_size=3,stride=1,padding=1,bias=False,
                                binarynum=binarynum,pooling=False,act=None)
        self.block3 = VggBlock_1w1aChannel(in_planes=256,planes=256,kernel_size=3,stride=1,padding=1,bias=False,
                                binarynum=binarynum,pooling=True,act=None)
        self.block4 = VggBlock_1w1aChannel(in_planes=256,planes=512,kernel_size=3,stride=1,padding=1,bias=False,
                               binarynum=binarynum,pooling=False,act=None)
        self.block5 = VggBlock_1w1aChannel(in_planes=512,planes=512,kernel_size=3,stride=1,padding=1,bias=False,
                                binarynum=binarynum,pooling=True,act=None)
        self.fc = nn.Linear(512*4*4, self.num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, Layer.BNNConv2d_1wnaChannel):
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
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



def vgg_small_cpba(pretrained=False, progress=True,**kwargs):
    model = VGG_SMALL_CPBA(**kwargs)
    return model

def vgg_small_1w1alayer(pretrained=False, progress=True,**kwargs):
    model = VGG_SMALL_1w1aLayer(**kwargs)
    return model

def vgg_small_1w1achannel(pretrained=False, progress=True,**kwargs):
    model = VGG_SMALL_1w1aChannel(**kwargs)
    return model


def test():
    net = vgg_small_1w1achannel(num_classes=10)
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())
    # print(net)
    # for name,child in net.named_children():
    #     print(type(child),name)
    #     if name=="features":
    #         print(name,child[0])

if __name__ == '__main__':
    test()
