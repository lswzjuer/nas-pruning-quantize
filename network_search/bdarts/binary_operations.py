# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-10-13 23:41:53
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-11-26 19:47:53

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import binary_layers as Layer


OPS = {
  'none' : lambda C, stride, group,affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, group,affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, group,affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, group,affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'group_conv_3x3' : lambda C, stride, group,affine: BinaryGroupConv(C, C, 3, stride, 1,group=group,affine=affine),
  'group_conv_5x5' : lambda C, stride, group,affine: BinaryGroupConv(C, C, 5, stride, 2, group=group,affine=affine),
  'dil_group_conv_3x3' : lambda C, stride, group,affine: BinaryDilGroupConv(C, C, 3, stride, 2, 2, group=group, affine=affine),
  'dil_group_conv_5x5' : lambda C, stride, group,affine: BinaryDilGroupConv(C, C, 5, stride, 4, 2,group=group, affine=affine),
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
      nn.BatchNorm2d(C_in, affine=affine),
      Layer.Conv2d_1w1a(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
    )

  def forward(self, x):
    return self.op(x)



# class BinaryDilGroupConv(nn.Module):
#   def __init__(self, C_in, C_out, kernel_size, stride, padding,dilation,group,affine=True,options="D"):
#     super(BinaryDilGroupConv, self).__init__()

#     self.group = int(C_in/4)
#     self.bn_1 = nn.BatchNorm2d(C_in, affine=affine)
#     self.conv_1 = Layer.Conv2d_1w1a(C_in, C_in, kernel_size=kernel_size, 
#                                     stride=stride, padding=padding,dilation=dilation,
#                                      groups=self.group, bias=False)

#     self.bn_2 = nn.BatchNorm2d(C_in, affine=affine)
#     self.conv_2 = Layer.Conv2d_1w1a(C_in, C_out, kernel_size=1, padding=0, bias=False)

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
#     out = self.conv_1(self.bn_1(x))
#     out += self.shortcut(x)
#     #out = F.hardtanh(out)
#     x1 = out
#     out = self.conv_2(self.bn_2(out))
#     out += x1
#     #out = F.hardtanh(out)
#     return out


# class BinaryGroupConv(nn.Module):
#   def __init__(self, C_in, C_out, kernel_size, stride, padding, group,affine=True,options="D"):
#     super(BinaryGroupConv, self).__init__()
#     self.group = int(C_in/4)
#     self.bn_1 = nn.BatchNorm2d(C_in, affine=affine)
#     self.conv_1 = Layer.Conv2d_1w1a(C_in, C_in, kernel_size=kernel_size, 
#                                     stride=stride, padding=padding, groups=self.group, bias=False)

#     self.bn_2 = nn.BatchNorm2d(C_in, affine=affine)
#     self.conv_2 = Layer.Conv2d_1w1a(C_in, C_out, kernel_size=1, padding=0, bias=False)
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
#     out = self.conv_1(self.bn_1(x))
#     out += self.shortcut(x)
#     #out = F.hardtanh(out)
#     x1 = out
#     out = self.conv_2(self.bn_2(out))
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
    self.bn = nn.BatchNorm2d(C_in, affine=affine)
    self.conv_1 = Layer.Conv2d_1w1a(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = Layer.Conv2d_1w1a(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 

  def forward(self, x):
    out = self.bn(x)
    out = torch.cat([self.conv_1(out), self.conv_2(out[:,:,1:,1:])], dim=1)
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


class BinaryDilGroupConv(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding,dilation,group,affine=True,options="D"):
    super(BinaryDilGroupConv, self).__init__()
    assert C_in ==  C_out
    self.group = int(C_in/4)
    self.bn_1 = nn.BatchNorm2d(C_in, affine=affine)
    self.conv_1 = Layer.Conv2d_1w1a(C_in, C_out, kernel_size=kernel_size, 
                                    stride=stride, padding=padding,dilation=dilation,
                                     groups=self.group, bias=False)
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


  def forward(self, x):
    out = self.shuffle(self.conv_1(self.bn_1(x)))
    out += self.shortcut(x)
    #out = F.hardtanh(out)
    return out


class BinaryGroupConv(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding, group,affine=True,options="D"):
    super(BinaryGroupConv, self).__init__()
    self.group = int(C_in/4)
    self.bn_1 = nn.BatchNorm2d(C_in, affine=affine)
    self.conv_1 = Layer.Conv2d_1w1a(C_in, C_in, kernel_size=kernel_size, 
                                    stride=stride, padding=padding, groups=self.group, bias=False)
    self.shuffle = ShuffleBlock(self.group)

    self.shortcut = nn.Sequential()
    if stride != 1:
        if options == "A":
            self.shortcut = LambdaLayer(lambda x:
                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, cin//4, cin//4), "constant", 0))
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


  def forward(self, x):
    out = self.shuffle(self.conv_1(self.bn_1(x)))
    out += self.shortcut(x)
    #out = F.hardtanh(out)
    return out