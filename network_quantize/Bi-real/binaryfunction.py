# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-09-25 22:14:08
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-10-13 14:57:56

import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F




def safeSign(tensor):
    tensor=torch.sign(tensor)
    tensor[tensor==0]=1
    return tensor


def getScales(tensor):
    '''
    :param tensor:  get channel mean value tensor
    :return:
    '''
    # Conv2d
    if tensor.dim() == 4:
        scales = torch.mean(torch.abs(tensor),dim=[1,2,3],keepdim=True).detach()
        scales = scales.expand_as(tensor)
        return scales
    # Dense layer
    # Take the output dimension as the scaling factor to solve the dimension
    elif tensor.dim() == 2:
        scales = torch.mean(torch.abs(tensor),dim=0,keepdim=True).detach()
        scales = scales.expand_as(tensor)
        return scales
    else:
        NotImplementedError("Don`t support this layer !")


class BinaryWeightFunc(Function):
    """
    Binarizarion deterministic op with backprob.\n
    Forward :
    :math:    w_b  = a_w * sign(w)
    Backward : \n
    :math:`d w_b/d w = a_w *(1/(n*a_w)  + 1_{|w|=<1} )`
                     = a_w * sign(w) + 1/n
        (Add a constant to the gradient outside the range)
    """
    @staticmethod
    def forward(ctx, input):
        binput = safeSign(input)
        scales = getScales(input)
        ctx.save_for_backward(input,scales)
        return Wscales*binput

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output * d w_b/d w = grad_output *( a * (d sign(w) /d w) + 1/n)
        input,scales = ctx.saved_tensors
        mask1 = input < -1
        mask2 = input < 0
        mask3 = input < 1
        mask = 0 * mask1.type(torch.float32) + (input*2 + 2) * (1-mask1.type(torch.float32))
        mask = mask * mask2.type(torch.float32) + (-input*2 + 2) * (1-mask2.type(torch.float32))
        mask = mask * mask3.type(torch.float32) + 0 * (1- mask3.type(torch.float32))
        return grad_input * scales * mask



class BinaryFunc(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = safeSign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        mask1 = input < -1
        mask2 = input < 0
        mask3 = input < 1
        mask = 0 * mask1.type(torch.float32) + (input*2 + 2) * (1-mask1.type(torch.float32))
        mask = mask * mask2.type(torch.float32) + (-input*2 + 2) * (1-mask2.type(torch.float32))
        mask = mask * mask3.type(torch.float32) + 0 * (1- mask3.type(torch.float32))
        return grad_output * mask




if __name__ == '__main__':

    # testdata=[[
    #     [[2,2],
    #      [0.5,-0.5]],
    #     [[-2,-2],
    #      [0.4,-0.4]],
    #     [[-1,-1],
    #      [1,1]],
    # ]]
    # testdata=torch.tensor(testdata,requires_grad=True)
    testdata = torch.ones((3,1,2,2),requires_grad=True)
    testdata = testdata * torch.tensor([-2,-0.4,0.4]).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(testdata)
    print(testdata)
    output=BinaryFunc().apply(testdata)
    weight = torch.ones(output.size())
    grad = torch.autograd.grad(outputs=output,inputs=testdata,grad_outputs=weight)
    print(grad[0])

