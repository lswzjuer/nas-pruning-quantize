# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-09-26 16:53:06
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-10-11 02:00:55



# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-09-19 17:37:02
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-09-19 21:56:05

import torch 
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import bireal_layers as Layer



__all__ = ['vgg_small_1w1a','vgg_small_1w32a']



class VGG_SMALL_1W1A(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG_SMALL_1W1A, self).__init__()
        self.num_classes=num_classes
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.conv1 = Layer.BiRealConv2d_1w1a(128, 128, kernel_size=3, padding=1, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        # self.nonlinear = nn.ReLU(inplace=True)
        # none or htanh
        # self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 = Layer.BiRealConv2d_1w1a(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = Layer.BiRealConv2d_1w1a(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = Layer.BiRealConv2d_1w1a(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = Layer.BiRealConv2d_1w1a(512, 512, kernel_size=3, padding=1, bias=False)
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
            elif isinstance(m, Layer.BiRealConv2d_1w1a):
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
                m.bias.data.zero_()


    def forward(self, x):
        x1 = self.conv0(x)
        x1 = self.bn0(x1)
        x1 = x1 + x

        x2 = self.conv1(x1)
        x2 = self.pooling(x2)
        x2 = self.bn1(x2)
        x2 = x2 + self.pooling(x1)

        x3 = self.conv2(x2)
        x3 = self.bn2(x3)
        x3 = x3 + x2
 
        x4 = self.conv3(x3)
        x4 = self.pooling(x4)
        x4 = self.bn3(x4)
        x4 = x4 + self.pooling(x3)


        x5 = self.conv4(x4)
        x5 = self.bn4(x5)
        x5 = x5 + self.pooling(x4)


        x6 = self.conv5(x5)
        x6 = self.pooling(x6)
        x6 = self.bn5(x6)
        x6 = x6 + self.pooling(x5)

        # for tiny imagenet
        if self.num_classes==200:
            x6 = self.pooling(x6)
        x6 = x6.view(x6.size(0), -1)
        x6 = self.fc(x6)
        return x6




class VGG_SMALL_1W32A(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG_SMALL_1W32A, self).__init__()
        self.num_classes=num_classes
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.conv1 = Layer.XNORConv2d_1w32a(128, 128, kernel_size=3, padding=1, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.nonlinear = nn.ReLU(inplace=True)
        # self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 =  Layer.XNORConv2d_1w32a(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 =  Layer.XNORConv2d_1w32a(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 =  Layer.XNORConv2d_1w32a(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 =  Layer.XNORConv2d_1w32a(512, 512, kernel_size=3, padding=1, bias=False)
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
            elif isinstance(m,  Layer.XNORConv2d_1w32a):
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




def vgg_small_1w1a(pretrained=False, progress=True,**kwargs):
    model = VGG_SMALL_1W1A(**kwargs)
    return model



def vgg_small_1w32a(pretrained=False, progress=True,**kwargs):
    model = VGG_SMALL_1W32A(**kwargs)
    return model



def test():
    net = vgg_small_1w1a(num_classes=10)
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())
    print(net)
    # for name,child in net.named_children():
    #     print(type(child),name)
    #     if name=="features":
    #         print(name,child[0])

if __name__ == '__main__':
    test()
