# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-09-21 12:37:58
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-09-21 15:40:54



import torch 
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import ir_1w1a
import ir_1w32a
from HtanhChannel import alphaHtanhChannel
from HtanhLayer import alphaHtanhLayer



__all__ =['vgg_small_1w1a_ahtanhlayer','vgg_small_1w1a_ahtanhlayershared',
        'vgg_small_1w1a_ahtanhchannel','vgg_small_1w32a_ahtanh']


class VGG_SMALL_1W1A_AHTANHLayer(nn.Module):

    def __init__(self,num_classes=10):
        super(VGG_SMALL_1W1A_AHTANHLayer, self).__init__()
        self.num_classes=num_classes
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.nonlinear0 = alphaHtanhLayer()
        self.conv1 = ir_1w32a.IRConv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.nonlinear1 = alphaHtanhLayer()
        # self.nonlinear = nn.ReLU(inplace=True)
        # self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 = ir_1w32a.IRConv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.nonlinear2 = alphaHtanhLayer()
        self.conv3 = ir_1w32a.IRConv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.nonlinear3 = alphaHtanhLayer()
        self.conv4 = ir_1w32a.IRConv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.nonlinear4 = alphaHtanhLayer()
        self.conv5 = ir_1w32a.IRConv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.nonlinear5 = alphaHtanhLayer()
        self.fc = nn.Linear(512*4*4, self.num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, ir_1w32a.IRConv2d):
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

    def alpha_parameters(self):
        act_params=[]
        for pname, p in self.named_parameters():
            if "nonlinear" in pname:
                act_params.append(p)
        return act_params

    def other_parameters(self):
        other_parameters=[]
        for pname,p in self.named_parameters():
            if "nonlinear" not in pname:
                other_parameters.append(p)
        return other_parameters



    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.nonlinear0(x)
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.bn1(x)
        x = self.nonlinear1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.nonlinear2(x)
        x = self.conv3(x)
        x = self.pooling(x)
        x = self.bn3(x)
        x = self.nonlinear3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.nonlinear4(x)
        x = self.conv5(x)
        x = self.pooling(x)
        x = self.bn5(x)
        x = self.nonlinear5(x)
        # for tiny imagenet
        if self.num_classes==200:
            x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



class VGG_SMALL_1W1A_AHTANHChannel(nn.Module):

    def __init__(self,num_classes=10):
        super(VGG_SMALL_1W1A_AHTANHChannel, self).__init__()
        self.num_classes=num_classes
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.nonlinear0 = alphaHtanhChannel(input_channel=128)
        self.conv1 = ir_1w32a.IRConv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.nonlinear1 = alphaHtanhChannel(input_channel=128)
        # self.nonlinear = nn.ReLU(inplace=True)
        # self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 = ir_1w32a.IRConv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.nonlinear2 = alphaHtanhChannel(input_channel=256)
        self.conv3 = ir_1w32a.IRConv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.nonlinear3 = alphaHtanhChannel(input_channel=256)
        self.conv4 = ir_1w32a.IRConv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.nonlinear4 = alphaHtanhChannel(input_channel=512)
        self.conv5 = ir_1w32a.IRConv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.nonlinear5 = alphaHtanhChannel(input_channel=512)
        self.fc = nn.Linear(512*4*4, self.num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, ir_1w32a.IRConv2d):
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

    def alpha_parameters(self):
        act_params=[]
        for pname, p in self.named_parameters():
            if "nonlinear" in pname:
                act_params.append(p)
        return act_params

    def other_parameters(self):
        other_parameters=[]
        for pname,p in self.named_parameters():
            if "nonlinear" not in pname:
                other_parameters.append(p)
        return other_parameters

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.nonlinear0(x)
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.bn1(x)
        x = self.nonlinear1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.nonlinear2(x)
        x = self.conv3(x)
        x = self.pooling(x)
        x = self.bn3(x)
        x = self.nonlinear3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.nonlinear4(x)
        x = self.conv5(x)
        x = self.pooling(x)
        x = self.bn5(x)
        x = self.nonlinear5(x)
        # for tiny imagenet
        if self.num_classes==200:
            x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x




class VGG_SMALL_1W1A_AHTANHLayerShared(nn.Module):

    def __init__(self,num_classes=10):
        super(VGG_SMALL_1W1A_AHTANHLayerShared, self).__init__()
        self.num_classes=num_classes
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.conv1 = ir_1w32a.IRConv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.nonlinear = alphaHtanhLayer()
        # self.nonlinear = nn.ReLU(inplace=True)
        # self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 = ir_1w32a.IRConv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = ir_1w32a.IRConv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = ir_1w32a.IRConv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = ir_1w32a.IRConv2d(512, 512, kernel_size=3, padding=1, bias=False)
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
            elif isinstance(m, ir_1w32a.IRConv2d):
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

    def alpha_parameters(self):
        act_params=[]
        for pname, p in self.named_parameters():
            if "nonlinear" in pname:
                act_params.append(p)
        return act_params

    def other_parameters(self):
        other_parameters=[]
        for pname,p in self.named_parameters():
            if "nonlinear" not in pname:
                other_parameters.append(p)
        return other_parameters


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


class VGG_SMALL_1W32A_AHTANH(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG_SMALL_1W32A_AHTANH, self).__init__()
        self.num_classes=num_classes
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.conv1 = ir_1w32a.IRConv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.nonlinear = alphaHtanhLayer()
        # self.nonlinear = nn.ReLU(inplace=True)
        # self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 = ir_1w32a.IRConv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = ir_1w32a.IRConv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = ir_1w32a.IRConv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = ir_1w32a.IRConv2d(512, 512, kernel_size=3, padding=1, bias=False)
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
            elif isinstance(m, ir_1w32a.IRConv2d):
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

    def alpha_parameters(self):
        act_params=[]
        for pname, p in self.named_parameters():
            if "nonlinear" in pname:
                act_params.append(p)
        return act_params

    def other_parameters(self):
        other_parameters=[]
        for pname,p in self.named_parameters():
            if "nonlinear" not in pname:
                other_parameters.append(p)
        return other_parameters

        
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



def vgg_small_1w1a_ahtanhlayer(pretrained=False, progress=True,**kwargs):
    model = VGG_SMALL_1W1A_AHTANHLayer(**kwargs)
    return model

def vgg_small_1w1a_ahtanhlayershared(pretrained=False, progress=True,**kwargs):
    model = VGG_SMALL_1W1A_AHTANHLayerShared(**kwargs)
    return model

def vgg_small_1w1a_ahtanhchannel(pretrained=False, progress=True,**kwargs):
    model = VGG_SMALL_1W1A_AHTANHChannel(**kwargs)
    return model

def vgg_small_1w32a_ahtanh(pretrained=False, progress=True,**kwargs):
    model = vgg_small_1w32a_ahtanh(**kwargs)
    return model



