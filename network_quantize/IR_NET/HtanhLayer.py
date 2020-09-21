# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-09-20 20:16:21
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-09-21 14:15:26
import torch
import torch.nn as nn
from torch.autograd import Function



def alphahtan(input, alpha):
    '''
    HtanH(min=-a,max=+a) channel wise, clamp C channel
    '''
    return torch.clamp(input, min=-1*alpha, max=alpha)


class alphaHtanHFunc(Function):
    """docstring for alphaHtanHFunc"""

    @staticmethod
    def forward(ctx, input, alpha):
        '''
        :param ctx:
        :param input: N C H W     tensor
        :param alpha:   1        one parameter
        :return:
        '''
        ctx.save_for_backward(input, alpha)
        return  alphahtan(input, alpha.data.abs().item())


    @staticmethod
    def backward(ctx, grad_output):
        input, alpha = ctx.saved_tensors
        alpha = alpha.data.abs().expand_as(input).detach()
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
        grad_alpha = grad_alpha.sum().view(1)
        return grad_input , grad_alpha



class alphaHtanhLayer(nn.Module):
    """docstring for AphlaHtanh"""

    def __init__(self,inplace=False):
        super(alphaHtanhLayer, self).__init__()
        self.inplace = inplace
        # C
        self.alpha = nn.Parameter(torch.ones(1),requires_grad=True)

    def forward(self, input):
        '''
        :param input: N C H W
        :return:
        '''
        if input.device != self.alpha.device:
            self.alpha = self.alpha.to(input.device)
        alpha_input = alphaHtanHFunc().apply(input, self.alpha)
        return alpha_input








if __name__ == "__main__":
    # 2X10
    testdata=torch.tensor([-4,-3,-2,-1,-0.5,0.5,1,2,4],requires_grad=True)
    # ly=nn.Linear(10,5)
    # output = ly(testdata)
    # print(output)
    act=alphaHtanhLayer()
    output = act(testdata)
    print(output)
    weight = torch.ones(output.size())
    output.backward(weight)
    for name, p in act.named_parameters():
        print(p,p.grad)
    # weight = torch.ones(output.size())
    # grad = torch.autograd.grad(outputs=output,inputs=testdata,grad_outputs=weight)
    # print(grad[0])

    # grad_alpha = torch.sum(testdata,dim=0)
    # print(grad_alpha)