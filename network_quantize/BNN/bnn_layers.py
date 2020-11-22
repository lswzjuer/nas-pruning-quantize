# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-09-25 21:49:54
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-11-21 22:14:23

import torch
import torch.nn as nn
import torch.nn.functional as F
import binaryfunction



class BNNConv2d_1w1a(nn.Conv2d):
    '''
    bnn 1w1a conv2d layers
    '''
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(BNNConv2d_1w1a, self).__init__(in_channels,out_channels,
                                        kernel_size=kernel_size,stride=stride,padding=padding,
                                        dilation=dilation,groups=groups,bias=bias)

    def forward(self, input):
        binput = binaryfunction.BinaryFunc().apply(input)
        bweight = binaryfunction.BinaryFunc().apply(self.weight)
        output = F.conv2d(binput, bweight, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output



class BNNDense_1w1a(nn.Linear):
    def __init__(self,in_features,out_features,bias=False):
        super(BNNDense_1w1a, self).__init__(in_features=in_features,out_features=out_features,bias=bias)

    def forward(self, input):
        binput = binaryfunction.BinaryFunc().apply(input)
        bweight = binaryfunction.BinaryFunc().apply(self.weight)
        output = F.linear(input=binput, weight=bweight, bias=self.bias)
        return output



class BNNConv2d_1w32a(nn.Conv2d):
    '''
    bnn 1w1a conv2d layers
    '''
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(BNNConv2d_1w32a, self).__init__(in_channels,out_channels,
                                        kernel_size=kernel_size,stride=stride,padding=padding,
                                        dilation=dilation,groups=groups,bias=bias)

    def forward(self, input):
        binput = input
        bweight = binaryfunction.BinaryFunc().apply(self.weight)
        output = F.conv2d(binput, bweight, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output


class BNNDense_1w32a(nn.Linear):
    def __init__(self,in_features,out_features,bias=False):
        super(BNNDense_1w32a, self).__init__(in_features=in_features,out_features=out_features,bias=bias)

    def forward(self, input):
        binput = input
        bweight = binaryfunction.BinaryFunc().apply(self.weight)
        output = F.linear(input=binput, weight=bweight, bias=self.bias)
        return output




if __name__ == "__main__":

    testdata = torch.ones((3,1,3,3),requires_grad=True)
    testdata = testdata * torch.tensor([-2,-0.5,0.5]).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(testdata)

    conv1=BNNConv2d_1w1a(1,1,kernel_size=3,binarynum=3,stride=1,padding=1)
    output=conv1(testdata)

    weight = torch.ones(output.size())
    grad = torch.autograd.grad(outputs=output,inputs=testdata,grad_outputs=weight)
    print(grad[0])
