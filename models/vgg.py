# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-09-16 18:14:54
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-09-19 16:34:35

'''VGG11/13/16/19 in Pytorch.'''


import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, cfg,num_classes=10):
        super(VGG, self).__init__()
        self.cfg=cfg
        self.features = self._make_layers(self.cfg)
        self.glopool= nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.classifier = nn.Linear(512, num_classes)


    def forward(self, x):
        out = self.features(x)
        out = self.glopool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]

        return nn.Sequential(*layers)




def VGG11(pretrained=False, progress=True, **kwargs):
    return VGG(cfg["VGG11"], **kwargs)



def VGG13(pretrained=False, progress=True, **kwargs):
    return VGG(cfg["VGG13"], **kwargs)


def VGG16(pretrained=False, progress=True, **kwargs):
    return VGG(cfg["VGG16"], **kwargs)


def VGG19(pretrained=False, progress=True, **kwargs):
    return VGG(cfg["VGG19"], **kwargs)


def test():
    net = VGG13(num_classes=10)
    x = torch.randn(2,3,64,64)
    y = net(x)
    print(y.size())
    # print(net)
    # for name,child in net.named_children():
    #     print(type(child),name)
    #     if name=="features":
    #         print(name,child[0])


if __name__ == '__main__':
    test()
