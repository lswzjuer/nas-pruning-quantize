# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-09-25 21:49:54
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-09-30 00:19:01



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




class BNNConv2d_1wnaChannel(nn.Conv2d):
    '''
    bnn 1w1a conv2d layers
    '''
    def __init__(self,in_channels, out_channels, kernel_size,binarynum=1,
                     stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(BNNConv2d_1wnaChannel, self).__init__(in_channels,out_channels,
                                        kernel_size=kernel_size,stride=stride,padding=padding,
                                        dilation=dilation,groups=groups,bias=bias)
        self.binarynum = binarynum
        self.shiftalphas = nn.Parameter(torch.zeros(size=(binarynum,in_channels)).to(torch.float32),requires_grad=True)
        self.betas = nn.Parameter(torch.ones(binarynum).to(torch.float32),requires_grad=True)

    def forward(self, input):
        # muti binary activate
        # cinput = clip(input + v,0,1)
        # binput = safesign(cinput - 0.5) (相当于 cinput 取 round(0,1),再*2-1 (-1,1))
        if self.shiftalphas.device != input.device:
            self.shiftalphas = self.shiftalphas.to(input.device)
            self.betas = self.betas.to(input.device)
        bweight = binaryfunction.BinaryFunc().apply(self.weight)
        outputs = []
        for index in range(self.binarynum):
            #input: N C H W    C--> 1 C 1 1      C:  binary_num C
            sinput = input + self.shiftalphas[index].unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(input)
            cinput = torch.clamp(sinput,min=0,max=1)
            binput = binaryfunction.BinaryFunc().apply(cinput-0.5)
            boutput = F.conv2d(binput, bweight, self.bias,
                              self.stride, self.padding,
                              self.dilation, self.groups)
            # N C H` W`   *  beta[index]        Beta: binary_num   
            #                                    这里应该修改成 binary num的形状减少计算量，也就是每个二值化基一个因子就可
            boutput = boutput * self.betas[index]
            outputs.append(boutput)
        # BNUM N C H` W`
        output = torch.sum(torch.stack(outputs,0),dim=0,keepdim=False)
        return output


class BNNConv2d_1wnaLayer(nn.Conv2d):
    '''
    bnn 1w1a conv2d layers
    '''
    def __init__(self,in_channels, out_channels, kernel_size,binarynum=1,
                     stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(BNNConv2d_1wnaLayer, self).__init__(in_channels,out_channels,
                                        kernel_size=kernel_size,stride=stride,padding=padding,
                                        dilation=dilation,groups=groups,bias=bias)
        self.binarynum = binarynum
        self.shiftalphas = nn.Parameter(torch.zeros(self.binarynum).to(torch.float32),requires_grad=True)
        self.betas = nn.Parameter(torch.ones(self.binarynum).to(torch.float32),requires_grad=True)

    def forward(self, input):
        # muti binary activate
        # cinput = clip(input + v,0,1)
        # binput = safesign(cinput - 0.5) (相当于 cinput 取 round(0,1),再*2-1 (-1,1))
        if self.shiftalphas.device != input.device:
            self.shiftalphas = self.shiftalphas.to(input.device)
            self.betas = self.betas.to(input.device)
        bweight = binaryfunction.BinaryFunc().apply(self.weight)
        outputs = []
        for index in range(self.binarynum):
            sinput = input + self.shiftalphas[index]
            cinput = torch.clamp(sinput,min=0,max=1)
            binput = binaryfunction.BinaryFunc().apply(cinput-0.5)
            boutput = F.conv2d(binput, bweight, self.bias,
                              self.stride, self.padding,
                              self.dilation, self.groups)
            boutput = boutput * self.betas[index]
            outputs.append(boutput)
        # BNUM N C H` W`
        output = torch.sum(torch.stack(outputs,0),dim=0,keepdim=False)
        return output



if __name__ == "__main__":

    testdata = torch.ones((3,1,3,3),requires_grad=True)
    testdata = testdata * torch.tensor([-2,-0.5,0.5]).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(testdata)

    conv1=BNNConv2d_1wnaChannel(1,1,kernel_size=3,binarynum=3,stride=1,padding=1)
    output=conv1(testdata)

    weight = torch.ones(output.size())
    grad = torch.autograd.grad(outputs=output,inputs=testdata,grad_outputs=weight)
    print(grad[0])
