# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-09-25 21:49:54
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-11-28 20:12:55

import torch
import torch.nn as nn
import torch.nn.functional as F
import binaryfunction
import math



class Conv2d_1w1a(nn.Conv2d):
    '''
    bnn 1w1a conv2d layers
    '''
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d_1w1a, self).__init__(in_channels,out_channels,
                                        kernel_size=kernel_size,stride=stride,padding=padding,
                                        dilation=dilation,groups=groups,bias=bias)

    def forward(self, input):

        # first 
        # bw = binaryfunction.BinaryFunc().apply(self.weight)
        # scale_b = torch.mean(torch.abs(self.weight),dim=[1,2,3],keepdim=True).detach()
        # bw = bw * scale_b
        # second

        w = self.weight
        # bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        # bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        # sw = torch.pow(torch.tensor([2]*bw.size(0)).to(input.device).float(), (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(bw.size(0), 1, 1, 1).detach()
        # bw = binaryfunction.BinaryFuncv2().apply(bw)
        # bw = bw * sw
        binput = binaryfunction.BinaryFuncv2().apply(input)
        output = F.conv2d(binput, w, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output



class Dense_1w1a(nn.Linear):
    def __init__(self,in_features,out_features,bias=False):
        super(Dense_1w1a, self).__init__(in_features=in_features,out_features=out_features,bias=bias)

    def forward(self, input):
        binput = binaryfunction.BinaryFunc().apply(input)
        bweight = binaryfunction.BinaryFunc().apply(self.weight)
        output = F.linear(input=binput, weight=bweight, bias=self.bias)
        return output



class Conv2d_1w32a(nn.Conv2d):
    '''
    bnn 1w1a conv2d layers
    '''
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d_1w32a, self).__init__(in_channels,out_channels,
                                        kernel_size=kernel_size,stride=stride,padding=padding,
                                        dilation=dilation,groups=groups,bias=bias)

    def forward(self, input):
        # first 
        # bw = binaryfunction.BinaryFunc().apply(self.weight)
        # scale_b = torch.mean(torch.abs(self.weight),dim=[1,2,3],keepdim=True).detach()
        # bw = bw * scale_b
        # second

        w = self.weight
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        sw = torch.pow(torch.tensor([2]*bw.size(0)).to(input.device).float(), (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(bw.size(0), 1, 1, 1).detach()
        bw = binaryfunction.BinaryFuncv2().apply(bw)
        bw = bw * sw
        binput = input
        output = F.conv2d(binput, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output


class Dense_1w32a(nn.Linear):
    def __init__(self,in_features,out_features,bias=False):
        super(Dense_1w32a, self).__init__(in_features=in_features,out_features=out_features,bias=bias)

    def forward(self, input):
        binput = input
        bweight = binaryfunction.BinaryFunc().apply(self.weight)
        output = F.linear(input=binput, weight=bweight, bias=self.bias)
        return output




if __name__ == "__main__":

    # testdata = torch.ones((3,1,3,3),requires_grad=True)
    # testdata = testdata * torch.tensor([-2,-0.5,0.5]).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(testdata)

    # conv1=Conv2d_1w1a(1,1,kernel_size=3,stride=1,padding=1)
    # output=conv1(testdata)

    # weight = torch.ones(output.size())
    # grad = torch.autograd.grad(outputs=output,inputs=testdata,grad_outputs=weight)
    # print(grad[0])


    testdata = torch.randn(10)
    testdata[0] = 0
    testdata[5] = 0
    print(testdata)
    print(F.softmax(testdata))
