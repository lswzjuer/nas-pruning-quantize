# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-09-19 17:33:13
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-09-20 20:06:00

from torch.autograd import Function
import torch


def safeSign(tensor):
    result = torch.sign(tensor)
    result[result==0] = 1
    return result


class BinaryQuantizeIRNetF(Function):
    '''
    IR-NET base binary function: kHtanH(t) <--->sign
    '''
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = safeSign(input)
        #out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None


class BinaryQuantizeIRNetM(torch.nn.Module):
    '''
    IRNet module
    '''
    def __init__(self):
        super(BinaryQuantizeIRNetM, self).__init__()

    def forward(self,x,k,t):
        bx=BinaryQuantizeIRNetF.apply(x,k,t)
        return bx


class BinaryQuantizeReActNetF(Function):
    """ ReActNet func"""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = safeSign(input)
        #out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        mask1 = input < -1
        mask2 = input < 0
        mask3 = input < 1
        grad_input = 0 * mask1.type(torch.float32) + (input*2 + 2) * (1-mask1.type(torch.float32))
        grad_input = grad_input * mask2.type(torch.float32) + (-input*2 + 2) * (1-mask2.type(torch.float32))
        grad_input = grad_input * mask3.type(torch.float32) + 0 * (1- mask3.type(torch.float32))

        return grad_input



class BinaryQuantizeReActNetM(torch.nn.Module):
    '''
    ReActNet module
    '''
    def __init__(self):
        super(BinaryQuantizeReActNetM, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3
        return out




if __name__ == '__main__':
    print(torch.sign(torch.zeros((2,2))))