import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype




##############  首先构建一个完备的二值化超网的在一般的训练集上面将其训练收敛
##############  利用FISTA算法和对分支mask掩码m的L1正则稀疏化来达到m稀疏剪枝搜索的目的
##############  两个可搜索单元normal reduce，每个4个中间节点，这样可以学习不规则的采样可搜索cell

def _weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal(m.weight)


class MixedOp(nn.Module):

  def __init__(self, C, stride,group):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C,stride,group,False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, group,reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction
    # 前面是reduce cell,则需要将input进行通道变换C_prev_prev-->C的同时进行分辨率减半
    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = BinaryConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = BinaryConvBN(C_prev, C, 1, 1, 0, affine=False)
    # 四个中间节点
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    for i in range(self._steps):
      # 每个节点都跟它之前的节点存在一个mixoperations
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride,group)
        self._ops.append(op)

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)
    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      # tmp_list=[]
      # for j in range(2+i):
      #   tmp =self._ops[offset+j](start[j],weights[offset+j])
      #   tmp_list.append(tmp)
      # s = torch.sum(tmp_list)
      # offset = len(states)
      # states.append(s)
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)
    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, steps=4, multiplier=4, stem_multiplier=3,group=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._steps = steps
    self._multiplier = multiplier

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
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr,group, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    # 最终整个网络的输出通道为c_prev  N,Cpre,1,1
    self.bn =nn.BatchNorm1d(C_prev)
    self.classifier = nn.Linear(C_prev, num_classes)

    self.apply(_weights_init)
    self._initialize_alphas()

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)
    self.alphas_normal = nn.Parameter(torch.randn(k, num_ops),requires_grad=True)
    self.alphas_reduce = nn.Parameter(torch.randn(k, num_ops),requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def weight_parameters(self):
    return [param for name, param in self.named_parameters() if "alphas" not in name]

  def freeze_arch_parameters(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)
    self.alphas_normal = nn.Parameter(torch.ones(k, num_ops),requires_grad=False)
    self.alphas_reduce = nn.Parameter(torch.ones(k, num_ops),requires_grad=False)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]
    
  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new


  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        # weights = F.softmax(self.alphas_reduce, dim=-1)
        weights = self.alphas_reduce
      else:
        # weights = F.softmax(self.alphas_normal, dim=-1)
        weights = self.alphas_normal
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    out = out.view(out.size(0),-1)
    out = self.bn(out)
    logits = self.classifier(out)
    return logits


  def multi_parsev1(self,weights):
    sweights = F.softmax(weights, dim=-1).data.cpu().numpy()
    gene = []
    start = 0
    n = 2
    for i in range(self._steps):
      end = start + n
      OW = weights[start:end].copy()
      W = sweights[start:end].copy()
      # 每个节点保留两个对应softmax最大值最大的连接
      edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
      # 每个节点保留两个对应非零边的个数最大的连接
      # edges = sorted(range(i + 2), key=lambda x: sum( 1 for k in range(len(W[x])) if k !=PRIMITIVES.index('none') and W[x][k] == 0))[:2]
      # 该节点应该保留的两条边中 边j的情况
      for j in edges:
        for k in range(len(W[j])):
          if k != PRIMITIVES.index("none") and OW[j][k] != 0:
            gene.append((PRIMITIVES[k],j,2+i))
      start = end
      n += 1
    return gene

  def multi_parsev2(self,weights):
    weights = weights.data.cpu().numpy()
    gene = []
    start = 0
    n = 2
    for i in range(self._steps):
      end = start + n
      W = weights[start:end].copy()
      # 每个节点保留两个对应softmax最大值最大的连接
      # edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
      # 每个节点保留两个对应非零边的个数最大的连接
      edges = sorted(range(i + 2), key=lambda x: sum( 1 for k in range(len(W[x])) if k !=PRIMITIVES.index('none') and W[x][k] == 0))[:2]
      # 该节点应该保留的两条边中 边j的情况
      for j in edges:
        for k in range(len(W[j])):
          if k != PRIMITIVES.index("none") and W[j][k] != 0:
            gene.append((PRIMITIVES[k],j,2+i))
      start = end
      n += 1
    return gene


  def multi_parsev3(self,weights):
    weights = weights.data.cpu().numpy()
    gene = []
    start = 0
    n = 2
    for i in range(self._steps):
      end = start + n
      W = weights[start:end].copy()
      # 保留所有mixop的非零连接
      for j in range(len(W)):
        for k in range(len(W[j])):
          if k != PRIMITIVES.index("none") and W[j][k] != 0:
            gene.append((PRIMITIVES[k],j,2+i))
      start = end
      n += 1
    return gene

  def genotypev1(self):
    gene_normal = self.multi_parsev1(self.alphas_normal)
    gene_reduce = self.multi_parsev1(self.alphas_reduce)
    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

  def genotypev2(self):
    gene_normal = self.multi_parsev2(self.alphas_normal)
    gene_reduce = self.multi_parsev2(self.alphas_reduce)
    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

  def genotypev3(self):
    gene_normal = self.multi_parsev3(self.alphas_normal)
    gene_reduce = self.multi_parsev3(self.alphas_reduce)
    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype





if __name__ == '__main__':
  net = Network(8,10,4,2,2,3,1)
  # print(net)
  print(net.arch_parameters())
  # print(net.genotypev1())
  print(net.genotypev2())
  # print(net.genotypev3())
  for name,papram in net.named_parameters():
    print(name)

