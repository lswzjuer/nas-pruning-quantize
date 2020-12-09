# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-11-16 13:50:22
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-11-16 14:39:30


import torch as t
import torch.nn as nn 
import torch


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
        

class QuanConv2d(t.nn.Conv2d):
    def __init__(self, m: t.nn.Conv2d, wbit =32, abit=32):
        assert type(m) == t.nn.Conv2d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode)
        self.wbit = wbit
        self.abit = abit
        self.quan_w_fn = weight_quantize_fn(self.wbit)
        self.quan_a_fn = activation_quantize_fn(self.abit)
        self.weight = t.nn.Parameter(m.weight.detach())
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.bias.detach())

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        quantized_act = self.quan_a_fn(x)
        return t.nn.functional.conv2d(quantized_act, quantized_weight, self.bias,
                                      self.stride, self.padding,
                                      self.dilation, self.groups)

class QuanLinear(t.nn.Linear):
    def __init__(self, m: t.nn.Linear, wbit=32, abit=32):
        assert type(m) == t.nn.Linear
        super().__init__(m.in_features, m.out_features,
                         bias=True if m.bias is not None else False)
        self.wbit = wbit
        self.abit = abit
        self.quan_w_fn = weight_quantize_fn(self.wbit)
        self.quan_a_fn = activation_quantize_fn(self.abit)
        self.weight = t.nn.Parameter(m.weight.detach())
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.bias.detach())

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        quantized_act = self.quan_a_fn(x)
        return t.nn.functional.linear(quantized_act, quantized_weight, self.bias)



QuanModuleMapping = {
    nn.Conv2d: QuanConv2d,
    nn.Linear: QuanLinear
}
