import sys
import binary_genotypes
from graphviz import Digraph


def plot(genotype, filename):
  g = Digraph(
      format='png',
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  g.node("c_{k-2}", fillcolor='darkseagreen2')
  g.node("c_{k-1}", fillcolor='darkseagreen2')
  assert len(genotype) % 2 == 0
  steps = len(genotype) // 2

  for i in range(steps):
    g.node(str(i), fillcolor='lightblue')

  for i in range(steps):
    for k in [2*i, 2*i + 1]:
      op, j = genotype[k]
      if j == 0:
        u = "c_{k-2}"
      elif j == 1:
        u = "c_{k-1}"
      else:
        u = str(j-2)
      v = str(i)
      g.edge(u, v, label=op, fillcolor="gray")

  g.node("c_{k}", fillcolor='palegoldenrod')
  for i in range(steps):
    g.edge(str(i), "c_{k}", fillcolor="gray")

  g.render(filename, view=True)


if __name__ == '__main__':

  import os 

  genotype_name_list = [
                        # "ic18_il5_node4_gc6convshuffleres_32w1a_drop245_archf10_nott_lr001_stage0",
                        # "ic18_il5_node4_gc6convshuffleres_32w1a_drop245_archf10_nott_lr001_stage1",
                        # "ic18_il5_node4_gc6convshuffleres_32w1a_drop245_archf10_nott_lr001_stage2",
                        
                        # "ic18_il5_node4_gc6convshuffleres_32w1a_drop245_archf10_nott_lr025_stage0",
                        # "ic18_il5_node4_gc6convshuffleres_32w1a_drop245_archf10_nott_lr025_stage1",
                        # "ic18_il5_node4_gc6convshuffleres_32w1a_drop245_archf10_nott_lr025_stage2",

                        # "ic18_il5_node4_gc6convshuffleres_32w1a_drop123_archf10_t02_lr025_stage0",
                        # "ic18_il5_node4_gc6convshuffleres_32w1a_drop123_archf10_t02_lr025_stage1",
                        # "ic18_il5_node4_gc6convshuffleres_32w1a_drop123_archf10_t02_lr025_stage2"

                        # "ic18_il5_node4_gc6convshuffleres_32w1a_nodrop_archf10_t02_lr025_stage0",
                        # "ic18_il5_node4_gc6convshuffleres_32w1a_nodrop_archf10_t02_lr025_stage1",
                        # "ic18_il5_node4_gc6convshuffleres_32w1a_nodrop_archf10_t02_lr025_stage2",

                        "ic18_il5_node4_gc6convshuffleres_32w1a_drop123_archf10_t02_lr025_adv5_stage0",
                        "ic18_il5_node4_gc6convshuffleres_32w1a_drop123_archf10_t02_lr025_adv5_stage1",
                        "ic18_il5_node4_gc6convshuffleres_32w1a_drop123_archf10_t02_lr025_adv5_stage2_15",
                        "ic18_il5_node4_gc6convshuffleres_32w1a_drop123_archf10_t02_lr025_adv5_stage2_19",
                        "ic18_il5_node4_gc6convshuffleres_32w1a_drop123_archf10_t02_lr025_adv5_stage2"
                          ]
  datapath="./cells/pbdarts"
  for i in range(len(genotype_name_list)):
    genotype_name = genotype_name_list[i]
    path = os.path.join(datapath,genotype_name)
    if not os.path.exists(path):
      os.mkdir(path)
    genotype = eval('binary_genotypes.{}'.format(genotype_name))
    plot(genotype.normal, os.path.join(path,"normal"))
    plot(genotype.reduce, os.path.join(path,"reduction"))
