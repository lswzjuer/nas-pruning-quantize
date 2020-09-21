# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-09-20 20:16:21
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-09-21 12:50:20


import torch
import torch.nn as nn
from torch.autograd import Function



def alphahtan(input, alpha):
    '''
    HtanH(min=-a,max=+a) channel wise, clamp C channel
    '''
    mask1 = input < -1*alpha
    mask2 = input <= alpha
    output = -1*alpha*mask1.type(torch.float32) + input * (1 - mask1.type(torch.float32))
    output = output * mask2.type(torch.float32) + alpha * (1 - mask2.type(torch.float32))
    return output


class alphaHtanHFunc(Function):
    """docstring for alphaHtanHFunc"""

    @staticmethod
    def forward(ctx, input, alpha):
        '''
        :param ctx:
        :param input: N C H W     tensor
        :param alpha:   C         parameter
        :return:
        '''
        alpha = alpha.data.abs().view(1, alpha.size(0), 1, 1).expand_as(input).detach()
        ctx.save_for_backward(input, alpha)
        return  alphahtan(input, alpha)


    @staticmethod
    def backward(ctx, grad_output):
        input, alpha = ctx.saved_tensors
        # grad_input
        mask1 = input <= alpha
        mask2 = input >= -alpha
        mask3 = mask1 * mask2
        grad_input = grad_output * (mask3.type(torch.float32))

        # grad alpha
        mask1 = input < -alpha
        mask2 = input <= alpha
        grad_alpha = -1 * mask1.type(torch.float32) + 0 * ( 1 - mask1.type(torch.float32))
        grad_alpha = grad_alpha * mask2.type(torch.float32) + 1 *(1 - mask2.type(torch.float32))
        grad_alpha = grad_output * grad_alpha
        # N C H W
        grad_alpha = grad_alpha.sum(dim=[0,2,3]).view(alpha.size(1))
        return grad_input , grad_alpha


class alphaHtanhChannel(nn.Module):
    """docstring for AphlaHtanh"""

    def __init__(self, input_channel=1, inplace=False):
        super(alphaHtanhChannel, self).__init__()
        self.input_channel = input_channel
        self.inplace = inplace
        assert self.input_channel is not None
        self.alpha = nn.Parameter(torch.ones(self.input_channel),requires_grad=True)


    def forward(self, input):
        '''
        :param input: N C H W
        :return:
        '''
        if input.device != self.alpha.device:
            self.alpha = self.alpha.to(input.device)
        assert self.input_channel == input.size(1), "The input channel is wrong !"
        return alphaHtanHFunc().apply(input, self.alpha)



if __name__ == "__main__":
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
    # ly=nn.Linear(10,5)
    # output = ly(testdata)
    # print(output)
    act=AphlaHtanhChannel(input_channel=3)
    output = act(testdata)
    print(output)
    weight = torch.ones(output.size())
    grad = torch.autograd.grad(outputs=output,inputs=testdata,grad_outputs=weight)
    print(grad[0])

