# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-09-26 17:02:17
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-10-11 02:25:49

import torch 
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import layers as Layer



class VGG_SMALL(nn.Module):
    def __init__(self, wbit=32,abit=32,num_classes=10):
        super(VGG_SMALL, self).__init__()
        self.num_classes=num_classes
        Conv2d = Layer.DoreafaConv2dv1(wbit,abit)
        self.conv0 = Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.conv1 = Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.nonlinear = nn.ReLU(inplace=True)
        # self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 = Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
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




def vgg_small(pretrained=False, progress=True,wbit=32,abit=32,**kwargs):
    model = VGG_SMALL(wbit=wbit,abit=abit,**kwargs)
    return model





def test():
    net = vgg_small(wbit=4,abit=4,num_classes=10)
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
