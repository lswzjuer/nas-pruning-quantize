# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-09-26 15:11:39
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-10-10 01:18:15

import torch 
import torch.nn as nn 
import torch.autograd.function as Function
import torch.nn.functional as F
import binaryfunction



class XNORConv2d_1w1a(nn.Conv2d):
    '''
    XNOR 1w1a conv2d layers
    '''
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(XNORConv2d_1w1a, self).__init__(in_channels,out_channels,
                                        kernel_size=kernel_size,stride=stride,padding=padding,
                                        dilation=dilation,groups=groups,bias=bias)

    def forward(self, input):

        # w_b = a * sign(w)
        bw = binaryfunction.BinaryFunc().apply(self.weight)
        scale_b = self.weight.abs().view(self.weight.size(0), -1).mean(-1).view(self.weight.size(0),1,1,1).detach()
        scale_bw = bw * scale_b
        # input_b = sign(input)
        binput = binaryfunction.BinaryFunc().apply(input)
        boutput = F.conv2d(binput, scale_bw,bias=self.bias,
                          stride=self.stride, padding=self.padding,
                          dilation=self.dilation,groups=self.groups)
        # compute output scale feature map ()
        # Equal to the scaling factor for each activation value of the convolution fast
        os = self.getScaleFeatureMap(input)
        os = os.detach()
        output = boutput * os
        return output


    def getScaleFeatureMap(self,input):
        # N C H W --> N 1 H W
        # Compute channel abs mean
        input_mean = torch.mean(torch.abs(input),dim=1,keepdim=True)
        # N 1 H W ---> N 1 H` W`
        # Calculate the plane mean for each convolution block by convolution
        kernel = torch.ones((1,1,self.kernel_size[0],self.kernel_size[1])).to(input.device)
        kernel.data.mul_(1 / (self.kernel_size[0] *self.kernel_size[1]))
        input_mean = F.conv2d(input_mean, kernel, 
                              stride=self.stride, padding=self.padding,
                              dilation=self.dilation,groups=self.groups)
        return input_mean



class XNORDense_1w1a(nn.Linear):
    def __init__(self,out_features,bias=True):
        super(XNORDense_1w1a, self).__init__(out_features=out_features,bias=bias)

    def forward(self, input):
        # w_b = a * sign(w)
        # dim(w)=DimIn*DimOUT  dim(a)= 1*DimOUT
        bw = binaryfunction.BinaryFunc().apply(self.weight)
        scale_b = self.weight.abs().mean(0).view(1,self.weight.size(1)).detach()
        scale_bw = bw * scale_b
        # input_b = sign(input)
        # dim(input) = N*DimIn
        binput = binaryfunction.BinaryFunc().apply(input)
        # dim(a_input) = Nx1
        si = torch.mean(torch.abs(input),dim=1,keepdim=True).detach()
        scale_binput = binput * si
        output = F.linear(input=scale_binput, weight=scale_bw, bias=self.bias)
        return output




class XNORConv2d_1w32a(nn.Conv2d):
    '''
    XNOR 1w32a conv2d layers
    '''
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(XNORConv2d_1w32a, self).__init__(in_channels,out_channels,
                                        kernel_size=kernel_size,stride=stride,padding=padding,
                                        dilation=dilation,groups=groups,bias=bias)

    def forward(self, input):

        # w_b = a * sign(w)
        bw = binaryfunction.BinaryFunc().apply(self.weight)
        scale_b = self.weight.abs().view(self.weight.size(0), -1).mean(-1).view(self.weight.size(0),1,1,1).detach()
        scale_bw = bw * scale_b
        boutput = F.conv2d(input, scale_bw,bias=self.bias,
                          stride=self.stride, padding=self.padding,
                          dilation=self.dilation,groups=self.groups)

        return boutput



class XNORDense_1w32a(nn.Linear):
    def __init__(self,out_features,bias=True):
        super(XNORDense_1w32a, self).__init__(out_features=out_features,bias=bias)

    def forward(self, input):
        # w_b = a * sign(w)
        # dim(w)=DimIn*DimOUT  dim(a)= 1*DimOUT
        bw = binaryfunction.BinaryFunc().apply(self.weight)
        scale_b = self.weight.abs().mean(0).view(1,self.weight.size(1)).detach()
        scale_bw = bw * scale_b
        output = F.linear(input=input, weight=scale_bw, bias=self.bias)
        return output



if __name__ == '__main__':

    testdata = torch.ones((3,1,3,3),requires_grad=True)
    testdata = testdata * torch.tensor([-2,-0.5,0.5]).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(testdata)

    conv1=XNORConv2d_1w1a(1,1,kernel_size=3,stride=1,padding=1)
    output=conv1(testdata)

    weight = torch.ones(output.size())
    grad = torch.autograd.grad(outputs=output,inputs=testdata,grad_outputs=weight)
    print(grad[0])




