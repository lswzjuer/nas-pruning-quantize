# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-10-10 23:03:51
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-10-10 23:57:32


import torch 
import torch.nn as nn 
from torch.autograd import Function

def safeSign(tensor):
    tensor=torch.sign(tensor)
    tensor[tensor==0]=1
    return tensor


class Quantize(Function):
    """docstring for Quantize"""
    @staticmethod
    def forward(ctx, input, kbit):
        if kbit == 32:
            out = input
        elif kbit == 1:
            out =safeSign(input)
        else:
            scale = float(2**kbit-1)
            out = torch.round(input * scale) / scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
      grad_input = grad_output.clone()
      return grad_input,None



