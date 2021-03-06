import torch
import torch.nn as nn
import torch.nn.functional as F
from binary_operations import *
from torch.autograd import Variable
from binary_genotypes import PRIMITIVES
from binary_genotypes import Genotype


class MixedOp(nn.Module):

  def __init__(self, C, stride,group,p):
    super(MixedOp, self).__init__()
    self.mix_ops = nn.ModuleList()
    self.p = p
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, group,False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      if isinstance(op, Identity) and p > 0:
          op = nn.Sequential(op, nn.Dropout(self.p))
      self.mix_ops.append(op)

  def update_p(self):
      for op in self.mix_ops:
          if isinstance(op, nn.Sequential):
              if isinstance(op[0], Identity):
                  op[1].p = self.p

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self.mix_ops))


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev,group=1,p=0.0):
    super(Cell, self).__init__()
    self.reduction = reduction
    self.p = p
    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = BinaryConv(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = BinaryConv(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self.cell_ops = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride, group, self.p)
        self.cell_ops.append(op)

  def update_p(self):
    for op in self.cell_ops:
      op.p = self.p
      op.update_p()

  def forward(self, s0, s1, weights):

    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self.cell_ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)




class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion=None, steps=4, multiplier=4, stem_multiplier=3,group=1,p=0.0):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self.p = p

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
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev,group=group,p=self.p)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  # def new(self):
  #   model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
  #   for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
  #       x.data.copy_(y.data)
  #   return model_new

  def update_p(self):
    for cell in self.cells:
      cell.p = self.p
      cell.update_p()


  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)
    self.alphas_normal = nn.Parameter(1e-3*torch.randn(k, num_ops), requires_grad=True)
    self.alphas_reduce = nn.Parameter(1e-3*torch.randn(k, num_ops), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def weight_parameters(self):
    network_params = []
    for k, v in self.named_parameters():
      if not (k.endswith('alphas_normal') or k.endswith('alphas_reduce')):
        network_params.append(v)
    return network_params


  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype




if __name__ == '__main__':
  net = Network(8,10,4,None,2,2,3,1,0.0)
  # print(net)
  print(net.arch_parameters())
  # print(net.genotypev1())
  print(net.genotype())
  # print(net.genotypev3())
  for name,papram in net.named_parameters():
    print(name)
  print(net.p)
  net.p = 0.1
  net.update_p()
  print(net.p)