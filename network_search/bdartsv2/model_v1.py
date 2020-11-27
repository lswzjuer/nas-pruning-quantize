import torch
import torch.nn as nn
from operations import *
from torch.autograd import Variable
from collections import Counter
from genotypes import Genotype
from genotypes import PRIMITIVES

def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


class MixedOp(nn.Module):
  def __init__(self, C,stride,group,operations_list):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for opname in operations_list:
      assert opname in PRIMITIVES
      op = OPS[opname](C,stride,group,True)
      if 'pool' in opname:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=True))
      self._ops.append(op)

  def forward(self, x, drop_prob):
    # self._ops 组成一个mixed-operations的操作集合
    output_list=[]
    if self.training and drop_prob > 0.:
      output_list=[]
      for op in self._ops:
        if not isinstance(op, Identity):
          output_list.append(drop_path(op(x), drop_prob))
        else:
          output_list.append(op(x))
      return sum(output_list)
    else:
      return sum(op(x) for op in self._ops)


class Cell(nn.Module):

  def __init__(self, genotype, C_prev_prev, C_prev, C, group, reduction, reduction_prev):
    super(Cell, self).__init__()
    print(C_prev_prev, C_prev, C)
    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C)
    else:
      self.preprocess0 = BinaryConvBN(C_prev_prev, C, 1, 1, 0)
    self.preprocess1 = BinaryConvBN(C_prev, C, 1, 1, 0)
    
    if reduction:
      op_names, starts, ends = zip(*genotype.reduce)
      concat = genotype.reduce_concat
    else:
      op_names, starts, ends = zip(*genotype.normal)
      concat = genotype.normal_concat

    self._compile(C,group, op_names, starts,ends, concat, reduction)

  def _compile(self, C, group, op_names, starts,ends, concat, reduction):
    assert len(op_names) ==  len(ends)
    nodes_num = len(set(ends))
    nodes_indexs_list=[[] for i in range(nodes_num)]
    for i in range(len(ends)):
      # 当前的序号start-->end end对应的中间节点序号是哪个
      j = ends[i] - 2
      nodes_indexs_list[j].append((op_names[i],starts[i]))
    # 分别处理nodes_num个中间节点对应的操作条目，将起始节点一直的条目合并为一个mixoperations
    operations_dict = {}
    for i in range(nodes_num):
      # 保存第i个中间节点所对应的运算情况 start: [opnames.....] 表示start--> 2+i节点间存在的若干个运算
      sub_operations_dict={}
      nodeslist = nodes_indexs_list[i]
      startindex = []
      for j in range(len(nodeslist)):
        startindex.append(nodeslist[j][1])
      startindex = set(startindex)
      for index in startindex:
        sub_operations_dict[index] = []
      for opname,start in nodeslist:
        sub_operations_dict[start].append(opname)
      operations_dict[i] = sub_operations_dict
    self._steps = nodes_num
    self._concat = concat
    self.multiplier = len(concat)
    # 建立所有的运算
    # 该列表包含nodes_num个nn.ModuleList()对象，
    # 每个对象都表示的是 第i个中间节点对应的所有的mixed-operations的列表
    # 列表里面的每一个元素都表示的是该中间节点的一个mixed-operation信息 [startnode,mixed/op]
    self.middleNode_list = [[] for i in range(nodes_num)]
    for mid_node in operations_dict.keys():
      # 中间节点mid_node对应的操作信息
      node_operations = operations_dict[mid_node]
      for startnode in node_operations.keys():
        # mixed-operation   startnode ---> midnode+2  操作为 operations_list
        operations_list = node_operations[startnode]
        assert len(operations_list) >= 1 
        # 如果是reduce cell 并且跟两个输入节点相连的mixed-operations需要stride=2
        stride = 2 if reduction and startnode <2 else 1 
        mixedop = MixedOp(C,stride,group,operations_list)
        self.middleNode_list[mid_node].append([startnode,mixedop])

  def forward(self, s0, s1, drop_prob):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)
    states = [s0, s1]
    assert self._steps == len(self,middleNode_list)
    for i in range(self._steps):
      # 第i个中间节点
      s = 0
      mixedops_list = self.middleNode_list[i]
      for inputnode, mixedop in mixedops_list:
        # 第i个中间节点的某一个mixed-operations的起始节点是inputnode,具体实现是mixedop
        assert inputnode < len(states)
        s += mixedop(states[inputnode],drop_prob)
      states += [s]
    return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR, self).__init__()
    self.features = nn.Sequential(
      nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
      Layer.Conv2d_1w1a(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      Layer.Conv2d_1w1a(128, 768, 2, bias=False),
      nn.BatchNorm2d(768),
    )
    self.bn =nn.BatchNorm1d(768)
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0),-1)
    x = self.bn(x)
    x = self.classifier(x)
    return x


class AuxiliaryHeadImageNet(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 14x14"""
    super(AuxiliaryHeadImageNet, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
      # Commenting it out for consistency with the experiments in the paper.
      # nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x


class NetworkCIFAR(nn.Module):

  def __init__(self, C, num_classes, layers,auxiliary, genotype, group=3):
    super(NetworkCIFAR, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary

    stem_multiplier = 3
    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
    
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, group,reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr
      if i == 2*layers//3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.bn =nn.BatchNorm1d(C_prev)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    logits_aux = None
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2*self._layers//3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    out = out.view(out.size(0),-1)
    out =self.bn(out)
    logits = self.classifier(out)
    return logits, logits_aux


class NetworkImageNet(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype):
    super(NetworkImageNet, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary

    self.stem0 = nn.Sequential(
      nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C // 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    self.stem1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    C_prev_prev, C_prev, C_curr = C, C, C

    self.cells = nn.ModuleList()
    reduction_prev = True
    for i in range(layers):
      if i in [layers // 3, 2 * layers // 3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
      if i == 2 * layers // 3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AvgPool2d(7)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    logits_aux = None
    s0 = self.stem0(input)
    s1 = self.stem1(s0)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2 * self._layers // 3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits, logits_aux


if __name__ == '__main__':
  gen1 = Genotype(normal=[('max_pool_3x3', 0, 2), ('avg_pool_3x3', 0, 2), ('skip_connect', 0, 2), ('group_conv_3x3', 0, 2), ('group_conv_5x5', 0, 2), ('dil_group_conv_3x3', 0, 2), ('dil_group_conv_5x5', 0, 2), ('max_pool_3x3', 1, 2), ('avg_pool_3x3', 1, 2), ('skip_connect', 1, 2), ('group_conv_3x3', 1, 2), ('group_conv_5x5', 1, 2), ('dil_group_conv_3x3', 1, 2), ('dil_group_conv_5x5', 1, 2), ('max_pool_3x3', 0, 3), ('avg_pool_3x3', 0, 3), ('skip_connect', 0, 3), ('group_conv_3x3', 0, 3), ('group_conv_5x5', 0, 3), ('dil_group_conv_3x3', 0, 3), ('dil_group_conv_5x5', 0, 3), ('max_pool_3x3', 1, 3), ('avg_pool_3x3', 1, 3), ('skip_connect', 1, 3), ('group_conv_3x3', 1, 3), ('group_conv_5x5', 1, 3), ('dil_group_conv_3x3', 1, 3), ('dil_group_conv_5x5', 1, 3), ('max_pool_3x3', 2, 3), ('avg_pool_3x3', 2, 3), ('skip_connect', 2, 3), ('group_conv_3x3', 2, 3), ('group_conv_5x5', 2, 3), ('dil_group_conv_3x3', 2, 3), ('dil_group_conv_5x5', 2, 3)], normal_concat=range(2, 4), reduce=[('max_pool_3x3', 0, 2), ('avg_pool_3x3', 0, 2), ('skip_connect', 0, 2), ('group_conv_3x3', 0, 2), ('group_conv_5x5', 0, 2), ('dil_group_conv_3x3', 0, 2), ('dil_group_conv_5x5', 0, 2), ('max_pool_3x3', 1, 2), ('avg_pool_3x3', 1, 2), ('skip_connect', 1, 2), ('group_conv_3x3', 1, 2), ('group_conv_5x5', 1, 2), ('dil_group_conv_3x3', 1, 2), ('dil_group_conv_5x5', 1, 2), ('max_pool_3x3', 0, 3), ('avg_pool_3x3', 0, 3), ('skip_connect', 0, 3), ('group_conv_3x3', 0, 3), ('group_conv_5x5', 0, 3), ('dil_group_conv_3x3', 0, 3), ('dil_group_conv_5x5', 0, 3), ('max_pool_3x3', 1, 3), ('avg_pool_3x3', 1, 3), ('skip_connect', 1, 3), ('group_conv_3x3', 1, 3), ('group_conv_5x5', 1, 3), ('dil_group_conv_3x3', 1, 3), ('dil_group_conv_5x5', 1, 3), ('max_pool_3x3', 2, 3), ('avg_pool_3x3', 2, 3), ('skip_connect', 2, 3), ('group_conv_3x3', 2, 3), ('group_conv_5x5', 2, 3), ('dil_group_conv_3x3', 2, 3), ('dil_group_conv_5x5', 2, 3)], reduce_concat=range(2, 4))
  gen2 = Genotype(normal=[('max_pool_3x3', 0, 2), ('avg_pool_3x3', 0, 2), ('skip_connect', 0, 2), ('group_conv_3x3', 0, 2), ('group_conv_5x5', 0, 2), ('dil_group_conv_3x3', 0, 2), ('dil_group_conv_5x5', 0, 2), ('max_pool_3x3', 1, 2), ('avg_pool_3x3', 1, 2), ('skip_connect', 1, 2), ('group_conv_3x3', 1, 2), ('group_conv_5x5', 1, 2), ('dil_group_conv_3x3', 1, 2), ('dil_group_conv_5x5', 1, 2), ('max_pool_3x3', 0, 3), ('avg_pool_3x3', 0, 3), ('skip_connect', 0, 3), ('group_conv_3x3', 0, 3), ('group_conv_5x5', 0, 3), ('dil_group_conv_3x3', 0, 3), ('dil_group_conv_5x5', 0, 3), ('max_pool_3x3', 1, 3), ('avg_pool_3x3', 1, 3), ('skip_connect', 1, 3), ('group_conv_3x3', 1, 3), ('group_conv_5x5', 1, 3), ('dil_group_conv_3x3', 1, 3), ('dil_group_conv_5x5', 1, 3)], normal_concat=range(2, 4), reduce=[('max_pool_3x3', 0, 2), ('avg_pool_3x3', 0, 2), ('skip_connect', 0, 2), ('group_conv_3x3', 0, 2), ('group_conv_5x5', 0, 2), ('dil_group_conv_3x3', 0, 2), ('dil_group_conv_5x5', 0, 2), ('max_pool_3x3', 1, 2), ('avg_pool_3x3', 1, 2), ('skip_connect', 1, 2), ('group_conv_3x3', 1, 2), ('group_conv_5x5', 1, 2), ('dil_group_conv_3x3', 1, 2), ('dil_group_conv_5x5', 1, 2), ('max_pool_3x3', 0, 3), ('avg_pool_3x3', 0, 3), ('skip_connect', 0, 3), ('group_conv_3x3', 0, 3), ('group_conv_5x5', 0, 3), ('dil_group_conv_3x3', 0, 3), ('dil_group_conv_5x5', 0, 3), ('max_pool_3x3', 1, 3), ('avg_pool_3x3', 1, 3), ('skip_connect', 1, 3), ('group_conv_3x3', 1, 3), ('group_conv_5x5', 1, 3), ('dil_group_conv_3x3', 1, 3), ('dil_group_conv_5x5', 1, 3)], reduce_concat=range(2, 4))

  net = NetworkCIFAR(8, 10, 4,True, gen1, group=1)
  print(net)