# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-10-10 23:04:14
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-11-16 14:08:10

import torch 
import torch.nn as nn
import torch.nn.functional as F
import binaryfunction

def uniform_quantize(k):
  class qfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
      if k == 32:
        out = input
      elif k == 1:
        out = torch.sign(input)
      else:
        n = float(2 ** k - 1)
        out = torch.round(input * n) / n
      return out

    @staticmethod
    def backward(ctx, grad_output):
      grad_input = grad_output.clone()
      return grad_input
      
  return qfn().apply

class weight_quantize_fn(nn.Module):
  def __init__(self, w_bit):
    super(weight_quantize_fn, self).__init__()
    assert w_bit <= 8 or w_bit == 32
    self.w_bit = w_bit
    self.uniform_q = uniform_quantize(k=w_bit)

  def forward(self, x):
    if self.w_bit == 32:
      weight_q = x
    elif self.w_bit == 1:
      E = torch.mean(torch.abs(x)).detach()
      weight_q = self.uniform_q(x / E) * E
    else:
      weight = torch.tanh(x)
      max_w = torch.max(torch.abs(weight)).detach()
      weight = weight / 2 / max_w + 0.5
      weight_q = max_w * (2 * self.uniform_q(weight) - 1)
    return weight_q


class activation_quantize_fn(nn.Module):
  def __init__(self, a_bit):
    super(activation_quantize_fn, self).__init__()
    assert a_bit <= 8 or a_bit == 32
    self.a_bit = a_bit
    self.uniform_q = uniform_quantize(k=a_bit)

  def forward(self, x):
    if self.a_bit == 32:
      activation_q = x
    else:
      activation_q = self.uniform_q(torch.clamp(x, 0, 1))
      # print(np.unique(activation_q.detach().numpy()))
    return activation_q


def DoreafaConv2dv1(wbit,abit):
    class DoreafaConv2dClass(nn.Conv2d):
        '''
        Doreafa quantized conv2d layers
        '''
        def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
            super(DoreafaConv2dClass, self).__init__(in_channels,out_channels,
                                            kernel_size=kernel_size,stride=stride,padding=padding,
                                            dilation=dilation,groups=groups,bias=bias)

            self.weight_func = weight_quantize_fn(wbit)
            self.activation_func = activation_quantize_fn(abit)

        def forward(self, input):
            qinput = self.activation_func(input)
            qweight = self.weight_func(self.weight)
            output = F.conv2d(qinput, qweight, self.bias,
                              self.stride, self.padding,
                              self.dilation, self.groups)
            return output

    return DoreafaConv2dClass




def DoreafaConv2dv2(wbit,abit):
    class DoreafaConv2dClass(nn.Conv2d):
        '''
        Doreafa quantized conv2d layers
        '''
        def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
            super(DoreafaConv2dClass, self).__init__(in_channels,out_channels,
                                            kernel_size=kernel_size,stride=stride,padding=padding,
                                            dilation=dilation,groups=groups,bias=bias)
            self.wbit = wbit
            self.abit = abit

        def forward(self, input):
            # quantize weight 
            if self.wbit == 32:
                qweight = self.weight
            elif self.wbit == 1:
                scale = torch.mean(torch.abs(self.weight)).detach()
                qweight = binaryfunction.Quantize(self.weight/scale,self.wbit) * scale
            else:
                # -max +max
                weight = torch.tanh(self.weight)
                max_w = torch.max(torch.abs(weight)).detach()
                # 0~1
                weight = weight / 2 / max_w + 0.5
                # -max +max
                qweight = max_w * (2 * binaryfunction.Quantize(weight,self.wbit) - 1)

            # quantize activation 
            if self.abit == 32:
                qinput = input
            # elif self.abit == 1:
            #     weight = binaryfunction.Quantize(input,self.abit)
            else:
                clamp_input = torch.clamp(input,0,1)
                qinput = binaryfunction.Quantize(clamp_input,self.abit)

            output = F.conv2d(qinput, qweight, self.bias,
                              self.stride, self.padding,
                              self.dilation, self.groups)
            return output
            
    return DoreafaConv2dClass