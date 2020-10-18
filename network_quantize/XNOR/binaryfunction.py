# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-09-25 22:14:08
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-10-18 00:29:46

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


class BinaryWeightFuncV1(Function):
    """
    Binarizarion deterministic op with backprob.\n
    Forward :
    :math:    w_b  = a_w * sign(w)
    Backward : \n
    :math:`d w_b/d w = a_w * 1_{|w|=<1}`
    """
    @staticmethod
    def forward(ctx, input):
        binput = safeSign(input)
        scales = getScales(input)
        ctx.save_for_backward(input,scales)
        return scales*binput

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output * d w_b/d w = grad_output * a * (d sign(w) /d w)
        input,scales = ctx.saved_tensors
        mask = torch.le(torch.abs(input),1.0).to(torch.float32)
        grad_input = grad_output * scales * mask
        return grad_input



class BinaryWeightFuncV2(Function):
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
        return scales*binput

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output * d w_b/d w = grad_output *( a * (d sign(w) /d w) + 1/n)
        input,scales = ctx.saved_tensors
        mask = torch.le(torch.abs(input),1.0).to(torch.float32)
        # conv2d
        if input.dim() == 4:
            channel_num = input.numel()/input.size(0)
        # dense
        else:
            channel_num = input.size(0)
        grad_input = grad_output * ( scales * mask + 1/channel_num)
        return grad_input


class BinaryActionFunc(Function):

    @staticmethod
    def forward(ctx,input):
        ctx.save_for_backward(input)
        binput = safeSign(input)
        return binput

    @staticmethod
    def backward(ctx,grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[torch.abs(input) > 1.001] = 0
        return grad_input


class BinaryFunc(Function):
    @staticmethod
    def forward(ctx,input):
        ctx.save_for_backward(input)
        binput = safeSign(input)
        return binput

    @staticmethod
    def backward(ctx,grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[torch.abs(input) > 1.001] = 0
        return grad_input


class BinaryFuncv2(Function):
    """
    Binarizarion deterministic op with backprob.\n
    Forward : \n
    :math:`r_b  = sign(r)`\n  tanh2x 
    Backward : \n
    :math:`d r_b/d r = d tanh2r / d r`
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = safeSign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = 2 * (1 - torch.pow(torch.tanh(input * 2), 2)) * grad_output
        return grad_input














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
    testdata = testdata * torch.tensor([-2,-0.5,0.5]).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(testdata)
    output=BinaryWeightFuncV2().apply(testdata)


    weight = torch.ones(output.size())
    grad = torch.autograd.grad(outputs=output,inputs=testdata,grad_outputs=weight)

