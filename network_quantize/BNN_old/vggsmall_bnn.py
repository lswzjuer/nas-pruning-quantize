# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-09-26 17:02:17
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-10-09 21:01:36

import torch 
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import bnn_layers as Layer



__all__ = ['vgg_small_1w32a_dense','vgg_small_1w1a_dense','vgg_small_1w1a_move',
            'vgg_small_1w1a','vgg_small_1w32a','vgg_small_cbap']



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

  


class VGG_SMALL_1W1A_CBAP(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG_SMALL_1W1A_CBAP, self).__init__()
        self.num_classes=num_classes
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.conv1 = Layer.BNNConv2d_1w1a(128, 128, kernel_size=3, padding=1, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        # self.nonlinear = nn.ReLU(inplace=True)
        # self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 = Layer.BNNConv2d_1w1a(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = Layer.BNNConv2d_1w1a(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = Layer.BNNConv2d_1w1a(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = Layer.BNNConv2d_1w1a(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512*4*4, self.num_classes)
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
        x = self.conv0(x)
        x = self.bn0(x)
        # x = self.nonlinear(x)

        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.nonlinear(x)
        x = self.pooling(x)

        x = self.conv2(x)
        x = self.bn2(x)
        # x = self.nonlinear(x)

        x = self.conv3(x)
        x = self.bn3(x)
        # x = self.nonlinear(x)
        x = self.pooling(x)

        x = self.conv4(x)
        x = self.bn4(x)
        # x = self.nonlinear(x)

        x = self.conv5(x)
        x = self.bn5(x)
        # x = self.nonlinear(x)
        x = self.pooling(x)

        # for tiny imagenet
        if self.num_classes==200:
            x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



class VGG_SMALL_1W1A(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG_SMALL_1W1A, self).__init__()
        self.num_classes=num_classes
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.conv1 = Layer.BNNConv2d_1w1a(128, 128, kernel_size=3, padding=1, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        # self.nonlinear = nn.ReLU(inplace=True)
        # self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 = Layer.BNNConv2d_1w1a(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = Layer.BNNConv2d_1w1a(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = Layer.BNNConv2d_1w1a(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = Layer.BNNConv2d_1w1a(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)

        self.fc = nn.Linear(512*4*4, self.num_classes)
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
        x = self.conv0(x)
        x = self.bn0(x)
        # x = self.nonlinear(x)

        x = self.conv1(x)
        x = self.pooling(x)
        x = self.bn1(x)
        # x = self.nonlinear(x)

        x = self.conv2(x)
        x = self.bn2(x)
        # x = self.nonlinear(x)

        x = self.conv3(x)
        x = self.pooling(x)
        x = self.bn3(x)
        # x = self.nonlinear(x)

        x = self.conv4(x)
        x = self.bn4(x)
        # x = self.nonlinear(x)

        x = self.conv5(x)
        x = self.pooling(x)
        x = self.bn5(x)
        # x = self.nonlinear(x)
        # for tiny imagenet
        if self.num_classes==200:
            x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



class VGG_SMALL_1W32A(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG_SMALL_1W32A, self).__init__()
        self.num_classes=num_classes
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.conv1 = Layer.BNNConv2d_1w32a(128, 128, kernel_size=3, padding=1, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.nonlinear = nn.ReLU(inplace=True)
        # self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 =  Layer.BNNConv2d_1w32a(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 =  Layer.BNNConv2d_1w32a(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 =  Layer.BNNConv2d_1w32a(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 =  Layer.BNNConv2d_1w32a(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512*4*4, self.num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m,  Layer.BNNConv2d_1w32a):
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
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.nonlinear(x)

        x = self.conv1(x)
        x = self.pooling(x)
        x = self.bn1(x)
        x = self.nonlinear(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.nonlinear(x)

        x = self.conv3(x)
        x = self.pooling(x)
        x = self.bn3(x)
        x = self.nonlinear(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.nonlinear(x)
        
        x = self.conv5(x)
        x = self.pooling(x)
        x = self.bn5(x)
        x = self.nonlinear(x)
        # for tiny imagenet
        if self.num_classes==200:
            x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x





class VGG_SMALL_1W1A_MOVE(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG_SMALL_1W1A_MOVE, self).__init__()
        self.num_classes=num_classes
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.move1 = MoveBlock(128)
        self.conv1 = Layer.BNNConv2d_1w1a(128, 128, kernel_size=3, padding=1, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        # self.nonlinear = nn.ReLU(inplace=True)
        # self.nonlinear = nn.Hardtanh(inplace=True)
        self.move2 = MoveBlock(128)
        self.conv2 = Layer.BNNConv2d_1w1a(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.move3 = MoveBlock(256)
        self.conv3 = Layer.BNNConv2d_1w1a(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.move4 = MoveBlock(256)
        self.conv4 = Layer.BNNConv2d_1w1a(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.move5 = MoveBlock(512)
        self.conv5 = Layer.BNNConv2d_1w1a(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)

        self.fc = nn.Linear(512*4*4, self.num_classes)
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
        x = self.conv0(x)
        x = self.bn0(x)
        # x = self.nonlinear(x)

        x = self.move1(x)
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.bn1(x)
        # x = self.nonlinear(x)

        x = self.move2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        # x = self.nonlinear(x)

        x = self.move3(x)
        x = self.conv3(x)
        x = self.pooling(x)
        x = self.bn3(x)
        # x = self.nonlinear(x)

        x = self.move4(x)
        x = self.conv4(x)
        x = self.bn4(x)
        # x = self.nonlinear(x)

        x = self.move5(x)
        x = self.conv5(x)
        x = self.pooling(x)
        x = self.bn5(x)
        # x = self.nonlinear(x)
        # for tiny imagenet
        if self.num_classes==200:
            x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x






class VGG_SMALL_1W1A_DENSE(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG_SMALL_1W1A_DENSE, self).__init__()
        self.num_classes=num_classes
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.conv1 = Layer.BNNConv2d_1w1a(128, 128, kernel_size=3, padding=1, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        # self.nonlinear = nn.ReLU(inplace=True)
        # self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 = Layer.BNNConv2d_1w1a(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = Layer.BNNConv2d_1w1a(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = Layer.BNNConv2d_1w1a(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = Layer.BNNConv2d_1w1a(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)

        self.fc =  Layer.BNNDense_1w1a(512*4*4, self.num_classes)
        self.bn6 = nn.BatchNorm1d(self.num_classes)
        self.scaleshift = ScaleAndShift(self.num_classes)
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
            elif isinstance(m, nn.Linear) or isinstance(m,Layer.BNNDense_1w1a):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        # x = self.nonlinear(x)

        x = self.conv1(x)
        x = self.pooling(x)
        x = self.bn1(x)
        # x = self.nonlinear(x)

        x = self.conv2(x)
        x = self.bn2(x)
        # x = self.nonlinear(x)

        x = self.conv3(x)
        x = self.pooling(x)
        x = self.bn3(x)
        # x = self.nonlinear(x)

        x = self.conv4(x)
        x = self.bn4(x)
        # x = self.nonlinear(x)

        x = self.conv5(x)
        x = self.pooling(x)
        x = self.bn5(x)
        # x = self.nonlinear(x)
        # for tiny imagenet
        if self.num_classes==200:
            x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # after binary scale and shift
        x = self.bn6(x)
        x = self.scaleshift(x)
        return x



class VGG_SMALL_1W32A_DENSE(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG_SMALL_1W32A_DENSE, self).__init__()
        self.num_classes=num_classes
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.conv1 = Layer.BNNConv2d_1w32a(128, 128, kernel_size=3, padding=1, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.nonlinear = nn.ReLU(inplace=True)
        # self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 =  Layer.BNNConv2d_1w32a(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 =  Layer.BNNConv2d_1w32a(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 =  Layer.BNNConv2d_1w32a(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 =  Layer.BNNConv2d_1w32a(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)


        self.fc =  Layer.BNNDense_1w32a(512*4*4, self.num_classes)
        self.bn6 = nn.BatchNorm1d(self.num_classes)
        self.scaleshift = ScaleAndShift(self.num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m,  Layer.BNNConv2d_1w32a):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) or isinstance(m,Layer.BNNDense_1w32a):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.nonlinear(x)

        x = self.conv1(x)
        x = self.pooling(x)
        x = self.bn1(x)
        x = self.nonlinear(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.nonlinear(x)

        x = self.conv3(x)
        x = self.pooling(x)
        x = self.bn3(x)
        x = self.nonlinear(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.nonlinear(x)
        
        x = self.conv5(x)
        x = self.pooling(x)
        x = self.bn5(x)
        x = self.nonlinear(x)
        # for tiny imagenet
        if self.num_classes==200:
            x = self.pooling(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        x = self.bn6(x)
        x = self.scaleshift(x)

        return x




def vgg_small_1w1a_cbap(pretrained=False, progress=True,**kwargs):
    model = VGG_SMALL_1W1A_CBAP(**kwargs)
    return model


def vgg_small_1w1a(pretrained=False, progress=True,**kwargs):
    model = VGG_SMALL_1W1A(**kwargs)
    return model


def vgg_small_1w32a(pretrained=False, progress=True,**kwargs):
    model = VGG_SMALL_1W32A(**kwargs)
    return model


def vgg_small_1w1a_move(pretrained=False, progress=True,**kwargs):
    model = VGG_SMALL_1W1A_MOVE(**kwargs)
    return model



def vgg_small_1w1a_dense(pretrained=False, progress=True,**kwargs):
    model = VGG_SMALL_1W1A_DENSE(**kwargs)
    return model


def vgg_small_1w32a_dense(pretrained=False, progress=True,**kwargs):
    model = VGG_SMALL_1W32A_DENSE(**kwargs)
    return model



def test():
    net = vgg_small_1w1a(num_classes=10)
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
