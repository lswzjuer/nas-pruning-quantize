# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-09-25 17:07:12
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-11-21 22:14:10


'''
BNN
Both the default input and the weight are fully accurate

'''

import torch 
from torch.autograd import Function

def safeSign(tensor):
    tensor=torch.sign(tensor)
    tensor[tensor==0]=1
    return tensor


class BinaryFunc(Function):
    """
    Binarizarion deterministic op with backprob.\n
    Forward : \n
    :math:`r_b  = sign(r)`\n
    Backward : \n
    :math:`d r_b/d r = 1_{|r|=<1}`
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = safeSign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
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
        grad_input = 3 * (1 - torch.pow(torch.tanh(input * 3), 2)) * grad_output
        return grad_input



if __name__ == '__main__':
    # test BinaryFunc
        # 1,3,2,2
    testdata=[[
        [[2,2],
         [0.5,-0.5]],
        [[-2,-2],
         [0.4,-0.4]],
        [[-1,-1],
         [1,1]],
    ]]
    testdata=torch.tensor(testdata,requires_grad=True)
    print(testdata,testdata.size())
    output=BinaryFunc().apply(testdata)
    print(output)
    weight = torch.ones(output.size())
    grad = torch.autograd.grad(outputs=output,inputs=testdata,grad_outputs=weight)
    print(grad[0])

