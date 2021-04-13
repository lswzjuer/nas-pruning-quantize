# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-10-13 23:41:53
# @Last Modified by:   liusongwei
# @Last Modified time: 2021-01-03 22:12:29

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import binary_layers as Layer


# OPS = {
#   'none' : lambda C, stride, group,affine: Zero(stride),
#   'avg_pool_3x3' : lambda C, stride, group,affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
#   'max_pool_3x3' : lambda C, stride, group,affine: nn.MaxPool2d(3, stride=stride, padding=1),
#   'skip_connect' : lambda C, stride, group,affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
#   'group_conv_3x3' : lambda C, stride, group,affine: BinaryGroupConv(C, C, 3, stride, 1,group=group,affine=affine),
#   'group_conv_5x5' : lambda C, stride, group,affine: BinaryGroupConv(C, C, 5, stride, 2, group=group,affine=affine),
#   'dil_group_conv_3x3' : lambda C, stride, group,affine: BinaryDilGroupConv(C, C, 3, stride, 2, 2, group=group, affine=affine),
#   'dil_group_conv_5x5' : lambda C, stride, group,affine: BinaryDilGroupConv(C, C, 5, stride, 4, 2,group=group, affine=affine),
# }


OPS = {
  'none' : lambda C, stride, group,affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, group,affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, group,affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, group,affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'group_conv_3x3' : lambda C, stride, group,affine: Reactblock(C, C, 3, stride, 1, 1, group=group, affine=affine),
  'group_conv_5x5' : lambda C, stride, group,affine: Reactblock(C, C, 5, stride, 2, 1,group=group, affine=affine),
  'dil_group_conv_3x3' : lambda C, stride, group,affine: Reactblock(C, C, 3, stride, 2, 2, group=group, affine=affine),
  'dil_group_conv_5x5' : lambda C, stride, group,affine: Reactblock(C, C, 5, stride, 4, 2,group=group, affine=affine),
}




class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)


class BinaryConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(BinaryConv, self).__init__()
    self.op = nn.Sequential(
      Layer.Conv2d_1w1a(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    out = self.op(x)
    return out



# class BinaryDilGroupConv(nn.Module):
#   def __init__(self, C_in, C_out, kernel_size, stride, padding,dilation,group,affine=True,options="B"):
#     super(BinaryDilGroupConv, self).__init__()

#     self.group = int(C_in/6)

#     self.conv_1 = Layer.Conv2d_1w1a(C_in, C_in, kernel_size=kernel_size, 
#                                     stride=stride, padding=padding,dilation=dilation,
#                                      groups=self.group, bias=False)
#     self.bn_1 = nn.BatchNorm2d(C_in, affine=affine)
#     self.conv_2 = Layer.Conv2d_1w1a(C_in, C_out, kernel_size=1, padding=0, bias=False)
#     self.bn_2 = nn.BatchNorm2d(C_out, affine=affine)

#     self.shortcut = nn.Sequential()
#     if stride != 1:
#         if options == "A":
#             self.shortcut = LambdaLayer(lambda x:
#                             F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, C_in//4, C_in//4), "constant", 0))
#         elif options == "B":
#             self.shortcut = nn.Sequential(
#                  Layer.Conv2d_1w1a(C_in, C_in, kernel_size=1, stride=stride, bias=False),
#                  nn.BatchNorm2d(C_in,affine=affine)
#             )
#         elif options == "C":
#             self.shortcut = nn.Sequential( 
#                 nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#             )
#         else:
#             self.shortcut = nn.Sequential( 
#                 nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
#             )


#   def forward(self, x):
#     out = self.bn_1(self.conv_1(x))
#     out += self.shortcut(x)
#     #out = F.hardtanh(out)
#     x1 = out
#     out = self.bn_2(self.conv_2(out))
#     out += x1
#     #out = F.hardtanh(out)
#     return out


# class BinaryGroupConv(nn.Module):
#   def __init__(self, C_in, C_out, kernel_size, stride, padding, group,affine=True,options="B"):
#     super(BinaryGroupConv, self).__init__()
#     self.group = int(C_in/6)
#     self.conv_1 = Layer.Conv2d_1w1a(C_in, C_in, kernel_size=kernel_size, 
#                                     stride=stride, padding=padding, groups=self.group, bias=False)
#     self.bn_1 = nn.BatchNorm2d(C_in, affine=affine)
#     self.conv_2 = Layer.Conv2d_1w1a(C_in, C_out, kernel_size=1, padding=0, bias=False)
#     self.bn_2 = nn.BatchNorm2d(C_out, affine=affine)

#     self.shortcut = nn.Sequential()
#     if stride != 1:
#         if options == "A":
#             self.shortcut = LambdaLayer(lambda x:
#                             F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, cin//4, cin//4), "constant", 0))
#         elif options == "B":
#             self.shortcut = nn.Sequential(
#                  Layer.Conv2d_1w1a(C_in, C_in, kernel_size=1, stride=stride, bias=False),
#                  nn.BatchNorm2d(C_in,affine=affine)
#             )
#         elif options == "C":
#             self.shortcut = nn.Sequential( 
#                 nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#             )
#         else:
#             self.shortcut = nn.Sequential( 
#                 nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
#             )


#   def forward(self, x):
#     out = self.bn_1(self.conv_1(x))
#     out += self.shortcut(x)
#     #out = F.hardtanh(out)
#     x1 = out
#     out = self.bn_2(self.conv_2(out))
#     out += x1
#     #out = F.hardtanh(out)
#     return out


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.bn = nn.BatchNorm2d(C_out, affine=affine)
    self.conv_1 = Layer.Conv2d_1w1a(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = Layer.Conv2d_1w1a(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 

  def forward(self, x):
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,C//g,H,W).permute(0,2,1,3,4).reshape(N,C,H,W)


# class BinaryDilGroupConv(nn.Module):
#   def __init__(self, C_in, C_out, kernel_size, stride, padding,dilation,group,affine=True,options="D"):
#     super(BinaryDilGroupConv, self).__init__()
#     assert C_in ==  C_out
#     #self.group = int(C_in/6)
#     self.group = 12
#     self.stride = stride
#     self.conv_1 = Layer.Conv2d_1w1a(C_in, C_out, kernel_size=kernel_size, 
#                                     stride=stride, padding=padding,dilation=dilation,
#                                      groups=self.group, bias=False)
#     self.bn_1 = nn.BatchNorm2d(C_in, affine=affine)
#     self.shuffle = ShuffleBlock(self.group)

#     self.shortcut = nn.Sequential()
#     if stride != 1:
#         if options == "A":
#             self.shortcut = LambdaLayer(lambda x:
#                             F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, C_in//4, C_in//4), "constant", 0))
#         elif options == "B":
#             self.shortcut = nn.Sequential(
#                  Layer.Conv2d_1w1a(C_in, C_in, kernel_size=1, stride=stride, bias=False),
#                  nn.BatchNorm2d(C_in,affine=affine)
#             )
#         elif options == "C":
#             self.shortcut = nn.Sequential(
#                 # nn.BatchNorm2d(C_in,affine=affine),
#                 nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#             )
#         else:
#             self.shortcut = nn.Sequential(
#                 # nn.BatchNorm2d(C_in,affine=affine),
#                 nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
#             )


#   def forward(self, x):
#     out = self.bn_1(self.conv_1(x))
#     out = self.shuffle(out)
#     out += self.shortcut(x)
#     # out = self.shuffle(out)
#     return out


# class BinaryGroupConv(nn.Module):
#   def __init__(self, C_in, C_out, kernel_size, stride, padding, group,affine=True,options="D"):
#     super(BinaryGroupConv, self).__init__()
#     #self.group = int(C_in/6)
#     self.group = 12
#     self.stride = stride
#     self.conv_1 = Layer.Conv2d_1w1a(C_in, C_in, kernel_size=kernel_size, 
#                                     stride=stride, padding=padding, groups=self.group, bias=False)
#     self.bn_1 = nn.BatchNorm2d(C_in, affine=affine)
#     self.shuffle = ShuffleBlock(self.group)
#     self.shortcut = nn.Sequential()
#     if stride != 1:
#         if options == "A":
#             self.shortcut = LambdaLayer(lambda x:
#                             F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, C_in//4, C_in//4), "constant", 0))
#         elif options == "B":
#             self.shortcut = nn.Sequential(
#                  Layer.Conv2d_1w1a(C_in, C_in, kernel_size=1, stride=stride, bias=False),
#                  nn.BatchNorm2d(C_in,affine=affine)
#             )
#         elif options == "C":
#             self.shortcut = nn.Sequential( 
#                 nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#             )
#         else:
#             self.shortcut = nn.Sequential( 
#                 nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
#             )


#   def forward(self, x):
#     out = self.bn_1(self.conv_1(x))
#     out = self.shuffle(out)
#     out += self.shortcut(x)
#     # out = self.shuffle(out)
#     return out




class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

class Reactblock(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding,dilation,group,affine=True,options="D"):
        super(Reactblock, self).__init__()
        norm_layer = nn.BatchNorm2d

        self.stride = stride
        self.inplanes = C_in
        self.planes = C_out
        self.group = int(C_in//6)

        # react sign
        self.move11 = LearnableBias(C_in)
        self.binary_3x3=nn.Conv2d(C_in, C_in, kernel_size=kernel_size, dilation=dilation,
                                    stride=stride, padding=padding, groups=self.group, bias=False)
        self.bn1 = norm_layer(C_in,affine=affine)
        self.shuffle = ShuffleBlock(self.group)
        self.shortcut = nn.Sequential()
        if stride != 1:
            if options == "A":
                self.shortcut = LambdaLayer(lambda x:
                                F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, C_in//4, C_in//4), "constant", 0))
            elif options == "B":
                self.shortcut = nn.Sequential(
                     Layer.Conv2d_1w1a(C_in, C_in, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(C_in,affine=affine)
                )
            elif options == "C":
                self.shortcut = nn.Sequential( 
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )
            else:
                self.shortcut = nn.Sequential( 
                    nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                )
        # react prelu
        self.move12 = LearnableBias(C_in)
        self.prelu1 = nn.PReLU(C_in)
        self.move13 = LearnableBias(C_in)

        # react sign
        self.move21 = LearnableBias(C_in)
        self.binary_pw = Layer.Conv2d_1w1a(C_in, C_out, kernel_size=1, stride=1, bias=False)
        self.bn2 = norm_layer(C_out,affine=affine)

        self.move22 = LearnableBias(C_out)
        self.prelu2 = nn.PReLU(C_out)
        self.move23 = LearnableBias(C_out)


    def forward(self, x):
        x1 = x
        out1 = self.move11(x)
        out1 = self.binary_3x3(out1)
        out1 = self.bn1(out1)
        out1 = self.shuffle(out1)
        out1 += self.shortcut(x1)

        out1 = self.move12(out1)
        out1 = self.prelu1(out1)
        out1 = self.move13(out1)

        out2 = self.move21(out1)
        out2 = self.binary_pw(out2)
        out2 = self.bn2(out2)
        out2 += out1

        out2 = self.move22(out2)
        out2 = self.prelu2(out2)
        out2 = self.move23(out2)

        return out2

