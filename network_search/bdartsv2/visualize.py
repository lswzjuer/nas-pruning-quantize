# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-10-14 17:09:30
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-10-14 17:37:37
import sys
import genotypes
from graphviz import Digraph

def plot(genotype, filename,steps=4):
    g = Digraph(
      format='pdf',
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
    g.body.extend(['rankdir=LR'])

    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')

    for i in range(steps):
        g.node(str(i), fillcolor='lightblue')
    for i in range(len(genotype)):
    # mixed-operations in start--->end(global index) 
    # end is mid nodes, assert end>=2 and  end = i+2, i is the middle index
        op,start,end = genotype[i]
        if start == 0:
            u = "c_{k-2}"
        elif start == 1:
            u = "c_{k-1}"
        else:
            # 对应的中间节点索引
            u = str(j-2)
        assert end>=2, " the end node index is wrong !"
        v = str(end -2)
        g.edge(u, v, label=op, fillcolor="gray")

    g.node("c_{k}", fillcolor='palegoldenrod')
    for i in range(steps):
        g.edge(str(i), "c_{k}", fillcolor="gray")
    g.render(filename, view=True)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
    sys.exit(1)
    genotype_name = sys.argv[1]
    try:
        genotype = eval('genotypes.{}'.format(genotype_name))
    except AttributeError:
        print("{} is not specified in genotypes.py".format(genotype_name)) 
        sys.exit(1)
    plot(genotype.normal, "normal")
    plot(genotype.reduce, "reduction")

