# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-12-11 16:45:12
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-12-11 16:56:32



import numpy as np 
from binary_genotypes  import PRIMITIVES
import binary_operations



def parse_gen(switches,weights):
    mixops = []
    assert len(switches) == len(weights)
    for i in range(len(switches)):
        mixop_array = weights[i]
        keep_idx = []
        for j in range(len(PRIMITIVES)):
            if switches[i][j]:
                keep_idx.append(j)
        assert len(keep_idx) == len(mixop_array)
        if switches[i][0]:
            mixop_array[0]=0
        max_value, max_index = float(np.max(mixop_array)), int(np.argmax(mixop_array))
        max_index_pri = keep_idx[max_index]
        max_op_name = PRIMITIVES[max_index_pri]
        assert max_op_name!='none'
        mixops.append((max_value,max_op_name))
    # get the final cell genotype based in normal_down_res
    n = 2
    start = 0
    mixops_gen=[]
    for i in range(4):
        end=start+n
        node_egdes=mixops[start:end].copy()
        keep_edges=sorted(range(2 + i), key=lambda x: -node_egdes[x][0])[:2]
        for j in keep_edges:
            op_name=node_egdes[j][1]
            mixops_gen.append((op_name,j))
        start=end
        n+=1
    return mixops_gen


if __name__ == '__main__':
    switches = []
    for i in range(14):
        switches.append([True for j in range(len(PRIMITIVES))])

    nromal = [[0.36193225 0.4379051  0.0161191  0.02069746 0.02797149 0.03840489
      0.02190726 0.07506248]
     [0.14874698 0.4382021  0.00555205 0.00456976 0.01933626 0.2138325
      0.05921678 0.11054359]
     [0.25426775 0.5625067  0.03110873 0.03069633 0.03765448 0.03093559
      0.01902944 0.0338011 ]
     [0.18767262 0.60414505 0.00812492 0.00974984 0.01632629 0.03341402
      0.05659105 0.08397621]
     [0.14125268 0.7334557  0.01533368 0.00646582 0.01605983 0.02090663
      0.00438534 0.06214031]
     [0.17075664 0.32027614 0.04807629 0.04313296 0.03628872 0.08493719
      0.13115026 0.16538177]
     [0.28165826 0.38778698 0.02194643 0.03327375 0.03618929 0.0749488
      0.08338676 0.08080978]
     [0.18221524 0.5563337  0.03731598 0.02879707 0.07023829 0.02768439
      0.02906159 0.06835366]
     [0.12437709 0.7878268  0.02589052 0.00713319 0.0063407  0.01732855
      0.01846603 0.01263708]
     [0.27924806 0.12657717 0.05513798 0.10103597 0.06939318 0.09558422
      0.10619463 0.16682878]
     [0.39360607 0.16481358 0.03322955 0.05694548 0.08762505 0.06722024
      0.07258383 0.12397628]
     [0.34780723 0.20023367 0.03774906 0.04975709 0.11062324 0.06319663
      0.08460145 0.10603157]
     [0.3756192  0.46067354 0.03725033 0.0136018  0.02810322 0.02580596
      0.03471563 0.02423032]
     [0.18215683 0.7414551  0.05356305 0.00316706 0.00615716 0.00508859
      0.00446206 0.00395018]

    parse_gen(switches,nromal)