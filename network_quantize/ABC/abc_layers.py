# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-09-25 21:49:54
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-09-27 21:17:24



import torch
import torch.nn as nn
import torch.nn.functional as F
import binaryfunction



class BNNConv2d_1w1a(nn.Conv2d):
    '''
    bnn 1w1a conv2d layers
    '''
    def __init__(self,in_channels, out_channels, kernel_size, 
                        stride=1, padding=0, dilation=1, groups=1, bias=True,
                        input_num=1):
        super(BNNConv2d_1w1a, self).__init__(in_channels,out_channels,
                                        kernel_size=kernel_size,stride=stride,padding=padding,
                                        dilation=dilation,groups=groups,bias=bias)
        self.shiftalpha=
        self.betas=


    def forward(self, input):
        #  get many input 
        binput = binaryfunction.BinaryFunc().apply(input)
        bweight = binaryfunction.BinaryFunc().apply(self.weight)
        outputs = []
        for binput in binputs:
            suboutput = F.conv2d(binput, bweight, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
            outputs.append(suboutput)
        output = torch.sum(torch.tensor(outputs) * self.betas,dim=0,keepdim=False)
        return output



class BNNDense_1w1a(nn.Linear):
    def __init__(self,out_features,bias):
        super(BNNDense_1w1a, self).__init__(out_features=out_features,bias=bias)

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
    def __init__(self,out_features,bias):
        super(BNNDense_1w32a, self).__init__(out_features=out_features,bias=bias)

    def forward(self, input):
        binput = input
        bweight = binaryfunction.BinaryFunc().apply(self.weight)
        output = F.linear(input=binput, weight=bweight, bias=self.bias)
        return output
