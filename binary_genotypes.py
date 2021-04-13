# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-10-13 23:39:15
# @Last Modified by:   liusongwei
# @Last Modified time: 2021-01-03 19:27:20

from collections import namedtuple
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'group_conv_3x3',
    'group_conv_5x5',
    'dil_group_conv_3x3',
    'dil_group_conv_5x5'
]



BATS = Genotype(normal=[('group_conv_5x5', 0), ('dil_group_conv_5x5', 1), 
                        ('max_pool_3x3', 0),('max_pool_3x3', 2),
                        ('group_conv_5x5', 0), ('dil_group_conv_5x5', 2), 
                        ('group_conv_3x3',0), ('dil_group_conv_5x5', 1)], 
                        normal_concat=range(2, 6), 
                reduce=[('max_pool_3x3', 0), ('group_conv_3x3', 1), 
                        ('group_conv_5x5', 0), ('avg_pool_3x3', 1), 
                        ('group_conv_5x5', 1), ('max_pool_3x3', 3), 
                        ('group_conv_5x5', 0), ('group_conv_5x5', 1)], 
                        reduce_concat=range(2, 6))



# BATS_DROP245_LR025_F10_ic18gc6convshuffleres_t02_stage0 = Genotype(normal=[('group_conv_3x3', 0), ('group_conv_3x3', 1), ('dil_group_conv_5x5', 2), ('group_conv_5x5', 0), ('dil_group_conv_5x5', 2), ('group_conv_5x5', 1), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 3), ('max_pool_3x3', 4)], reduce_concat=range(2, 6))
# BATS_DROP245_LR025_F10_ic18gc6convshuffleres_t02_adv_stage0 = Genotype(normal=[('dil_group_conv_5x5', 1), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 2), ('avg_pool_3x3', 0), ('dil_group_conv_5x5', 3), ('avg_pool_3x3', 0), ('dil_group_conv_5x5', 4), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('group_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_group_conv_5x5', 2), ('dil_group_conv_5x5', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))


# ic18_il5_node4_gc6convshuffleres_32w1a_drop245_archf10_nott_lr001
ic18_il5_node4_gc6convshuffleres_32w1a_drop245_archf10_nott_lr001_stage0 = Genotype(normal=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 3), ('max_pool_3x3', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))
ic18_il5_node4_gc6convshuffleres_32w1a_drop245_archf10_nott_lr001_stage1 = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('group_conv_5x5', 0), ('max_pool_3x3', 3), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))
ic18_il5_node4_gc6convshuffleres_32w1a_drop245_archf10_nott_lr001_stage2 = Genotype(normal=[('max_pool_3x3', 0), ('dil_group_conv_5x5', 1), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 1), ('dil_group_conv_5x5', 2), ('dil_group_conv_5x5', 1), ('dil_group_conv_5x5', 0), ('dil_group_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('dil_group_conv_5x5', 2), ('group_conv_3x3', 0), ('max_pool_3x3', 1), ('group_conv_5x5', 0)], reduce_concat=range(2, 6))

# ic18_il5_node4_gc6convshuffleres_32w1a_drop245_archf10_nott_lr025
ic18_il5_node4_gc6convshuffleres_32w1a_drop245_archf10_nott_lr025_stage0 = Genotype(normal=[('max_pool_3x3', 1), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_group_conv_3x3', 0), ('max_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_group_conv_5x5', 2), ('max_pool_3x3', 1), ('dil_group_conv_5x5', 2), ('dil_group_conv_5x5', 3), ('max_pool_3x3', 2), ('max_pool_3x3', 4)], reduce_concat=range(2, 6))
ic18_il5_node4_gc6convshuffleres_32w1a_drop245_archf10_nott_lr025_stage1 = Genotype(normal=[('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 1), ('group_conv_3x3', 1), ('dil_group_conv_5x5', 2), ('dil_group_conv_5x5', 0), ('max_pool_3x3', 1), ('group_conv_5x5', 1), ('dil_group_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('group_conv_5x5', 0), ('skip_connect', 1), ('max_pool_3x3', 3), ('dil_group_conv_5x5', 0)], reduce_concat=range(2, 6))
ic18_il5_node4_gc6convshuffleres_32w1a_drop245_archf10_nott_lr025_stage2 = Genotype(normal=[('group_conv_3x3', 0), ('dil_group_conv_5x5', 1), ('group_conv_3x3', 0), ('group_conv_5x5', 1), ('group_conv_5x5', 1), ('group_conv_3x3', 0), ('group_conv_5x5', 1), ('dil_group_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 4), ('skip_connect', 0)], reduce_concat=range(2, 6))


# ic18_il5_node4_gc6convshuffleres_32w1a_drop123_archf10_t02_lr025
ic18_il5_node4_gc6convshuffleres_32w1a_drop123_archf10_t02_lr025_stage0 = Genotype(normal=[('group_conv_5x5', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('group_conv_5x5', 0), ('dil_group_conv_5x5', 3), ('group_conv_5x5', 0), ('max_pool_3x3', 4), ('dil_group_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_group_conv_5x5', 2), ('max_pool_3x3', 1), ('dil_group_conv_5x5', 2), ('max_pool_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))
ic18_il5_node4_gc6convshuffleres_32w1a_drop123_archf10_t02_lr025_stage1 = Genotype(normal=[('group_conv_3x3', 1), ('group_conv_3x3', 0), ('dil_group_conv_5x5', 2), ('dil_group_conv_5x5', 1), ('dil_group_conv_5x5', 3), ('dil_group_conv_5x5', 0), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('skip_connect', 1)], reduce_concat=range(2, 6))
ic18_il5_node4_gc6convshuffleres_32w1a_drop123_archf10_t02_lr025_stage2 = Genotype(normal=[('group_conv_3x3', 0), ('group_conv_3x3', 1), ('dil_group_conv_5x5', 2), ('group_conv_5x5', 0), ('group_conv_5x5', 3), ('group_conv_5x5', 1), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('group_conv_5x5', 0), ('skip_connect', 1), ('skip_connect', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),('max_pool_3x3', 1), ('max_pool_3x3', 3), ('skip_connect', 1)], reduce_concat=range(2, 6))



# ic18_il5_node4_gc6convshuffleres_32w1a_nodrop_archf10_t02_lr025
ic18_il5_node4_gc6convshuffleres_32w1a_nodrop_archf10_t02_lr025_stage0 = Genotype(normal=[('group_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_group_conv_5x5', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 3), ('max_pool_3x3', 0), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_group_conv_5x5', 2), ('skip_connect', 0), ('skip_connect', 0), ('group_conv_5x5', 1), ('max_pool_3x3', 2), ('skip_connect', 0)], reduce_concat=range(2, 6))
ic18_il5_node4_gc6convshuffleres_32w1a_nodrop_archf10_t02_lr025_stage1 = Genotype(normal=[('max_pool_3x3', 1), ('group_conv_5x5', 0), ('dil_group_conv_5x5', 2), ('max_pool_3x3', 0), ('group_conv_5x5', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 4), ('group_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_group_conv_5x5', 2), ('skip_connect', 0), ('dil_group_conv_5x5', 3), ('max_pool_3x3', 2), ('dil_group_conv_5x5', 2), ('group_conv_5x5', 4)], reduce_concat=range(2, 6))
ic18_il5_node4_gc6convshuffleres_32w1a_nodrop_archf10_t02_lr025_stage2 = Genotype(normal=[('group_conv_5x5', 0), ('group_conv_5x5', 1), ('dil_group_conv_5x5', 1), ('dil_group_conv_5x5', 2), ('group_conv_5x5', 0), ('dil_group_conv_5x5', 3), ('group_conv_5x5', 3), ('max_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('group_conv_5x5', 0), ('max_pool_3x3', 2), ('skip_connect', 0), ('max_pool_3x3', 2), ('dil_group_conv_5x5', 1), ('group_conv_5x5', 2)], reduce_concat=range(2, 6))


# ic18_il5_node4_gc6convshuffleres_32w1a_drop123_archf10_t02_lr025_adv5
ic18_il5_node4_gc6convshuffleres_32w1a_drop123_archf10_t02_lr025_adv5_stage0 = Genotype(normal=[('dil_group_conv_5x5', 1), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 3), ('max_pool_3x3', 0), ('skip_connect', 0), ('dil_group_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('group_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_group_conv_5x5', 2), ('dil_group_conv_5x5', 1), ('group_conv_5x5', 1), ('max_pool_3x3', 2), ('group_conv_5x5', 0), ('skip_connect', 1)], reduce_concat=range(2, 6))
ic18_il5_node4_gc6convshuffleres_32w1a_drop123_archf10_t02_lr025_adv5_stage1 = Genotype(normal=[('dil_group_conv_5x5', 0), ('max_pool_3x3', 1), ('dil_group_conv_5x5', 2), ('max_pool_3x3', 0), ('group_conv_5x5', 3), ('dil_group_conv_3x3', 2), ('max_pool_3x3', 4), ('group_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_group_conv_5x5', 2), ('dil_group_conv_5x5', 1), ('group_conv_5x5', 1), ('dil_group_conv_5x5', 3), ('dil_group_conv_3x3', 1), ('group_conv_5x5', 0)], reduce_concat=range(2, 6))
ic18_il5_node4_gc6convshuffleres_32w1a_drop123_archf10_t02_lr025_adv5_stage2_15 = Genotype(normal=[('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 1), ('dil_group_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_group_conv_3x3', 1), ('dil_group_conv_5x5', 3), ('max_pool_3x3', 1), ('group_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('group_conv_3x3', 0), ('avg_pool_3x3', 1), ('group_conv_3x3', 0), ('group_conv_5x5', 1), ('group_conv_5x5', 2), ('skip_connect', 1), ('skip_connect', 0)], reduce_concat=range(2, 6))
ic18_il5_node4_gc6convshuffleres_32w1a_drop123_archf10_t02_lr025_adv5_stage2_19 = Genotype(normal=[('dil_group_conv_5x5', 1), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 1), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 3), ('dil_group_conv_3x3', 1), ('max_pool_3x3', 1), ('dil_group_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('group_conv_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('dil_group_conv_5x5', 2), ('skip_connect', 1), ('skip_connect', 0)], reduce_concat=range(2, 6))
ic18_il5_node4_gc6convshuffleres_32w1a_drop123_archf10_t02_lr025_adv5_stage2 = Genotype(normal=[('dil_group_conv_5x5', 0), ('max_pool_3x3', 1), ('dil_group_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_group_conv_5x5', 3), ('max_pool_3x3', 0), ('dil_group_conv_5x5', 4), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('group_conv_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('group_conv_5x5', 2), ('skip_connect', 1), ('group_conv_3x3', 2)], reduce_concat=range(2, 6))




# ic18_il5_node4_gc6convshuffleres_32w1a_nodrop_archf10_t02_lr025
ic18_il5_node4_gc6convshuffleres_32w1a_nodrop_archf10_t02_lr025_stage0_t2 =  Genotype(normal=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_group_conv_5x5', 2), ('dil_group_conv_5x5', 1), ('max_pool_3x3', 3), ('dil_group_conv_5x5', 2), ('max_pool_3x3', 4), ('group_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 3), ('avg_pool_3x3', 3), ('dil_group_conv_5x5', 4)], reduce_concat=range(2, 6))
ic18_il5_node4_gc6convshuffleres_32w1a_nodrop_archf10_t02_lr025_stage1_t2 = Genotype(normal=[('group_conv_5x5', 0), ('max_pool_3x3', 1), ('dil_group_conv_5x5', 2), ('dil_group_conv_5x5', 0), ('max_pool_3x3', 3), ('group_conv_5x5', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('group_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 1)], reduce_concat=range(2, 6))
ic18_il5_node4_gc6convshuffleres_32w1a_nodrop_archf10_t02_lr025_stage2_t2 = Genotype(normal=[('dil_group_conv_5x5', 1), ('group_conv_5x5', 0), ('dil_group_conv_5x5', 2), ('dil_group_conv_5x5', 0), ('group_conv_5x5', 0), ('max_pool_3x3', 3), ('max_pool_3x3', 3), ('group_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 1)], reduce_concat=range(2, 6))


#  ic18_il5_node4_gc6convshuffleres_32w1a adv2 drop123 nott 
drop123_nott_adv2_stage0 = Genotype(normal=[('max_pool_3x3', 0), ('skip_connect', 1), ('group_conv_5x5', 0), ('avg_pool_3x3', 1), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 1), ('dil_group_conv_5x5', 0), ('group_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('dil_group_conv_5x5', 2), ('group_conv_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 1), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))
drop123_nott_adv2_stage1 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('dil_group_conv_5x5', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 4), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))
drop123_nott_adv2_stage2 = Genotype(normal=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_group_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_group_conv_5x5', 1), ('dil_group_conv_5x5', 1), ('group_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('group_conv_3x3', 2), ('dil_group_conv_5x5', 3), ('avg_pool_3x3', 0), ('skip_connect', 4)], reduce_concat=range(2, 6))

# reactnet  drop123 nott 
reactnet_drop123_nott_stage0 = Genotype(normal=[('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 1), ('dil_group_conv_5x5', 2), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 2), ('dil_group_conv_5x5', 0), ('group_conv_3x3', 0), ('group_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_group_conv_5x5', 3), ('dil_group_conv_5x5', 2), ('dil_group_conv_5x5', 3), ('dil_group_conv_5x5', 2)], reduce_concat=range(2, 6))
reactnet_drop123_nott_stage1 = Genotype(normal=[('group_conv_5x5', 0), ('group_conv_3x3', 1), ('dil_group_conv_3x3', 1), ('skip_connect', 0), ('dil_group_conv_5x5', 1), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 1), ('group_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('dil_group_conv_5x5', 2), ('dil_group_conv_5x5', 3), ('skip_connect', 2), ('dil_group_conv_5x5', 4)], reduce_concat=range(2, 6))
reactnet_drop123_nott_stage2 = Genotype(normal=[('group_conv_5x5', 0), ('dil_group_conv_5x5', 1), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 1), ('dil_group_conv_5x5', 1), ('dil_group_conv_3x3', 3), ('dil_group_conv_5x5', 1), ('dil_group_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 1), ('dil_group_conv_5x5', 2), ('group_conv_5x5', 0), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))



# reactnet adv2 drop123 nott 
reactnet_drop123_nott_stage0_att2 = Genotype(normal=[('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 1), ('group_conv_5x5', 2), ('group_conv_3x3', 0), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0), ('dil_group_conv_5x5', 4)], reduce_concat=range(2, 6))





# # group 5 倾向的分布
# [[3.73794049e-01 6.96556792e-02 4.23684567e-02 5.10031283e-02
#   1.05683692e-01 1.43453777e-01 5.46215735e-02 1.59419626e-01]  
#  [8.81329417e-01 1.40117370e-02 9.13789682e-03 1.91913098e-02
#   1.78665090e-02 1.87814813e-02 1.20388847e-02 2.76427623e-02]
#  [9.81759548e-01 9.94532602e-04 4.78425936e-04 1.02681795e-03
#   4.56384895e-03 2.23650038e-03 2.54966132e-03 6.39066985e-03]
#  [9.41083252e-01 4.89461760e-04 1.74149711e-04 4.07241518e-04
#   1.74484018e-03 2.27629337e-02 4.30680346e-03 2.90313903e-02]
#  [9.66595292e-01 5.54866507e-04 2.40008623e-04 1.18526659e-04
#   7.82217074e-04 6.09822571e-03 4.21452802e-03 2.13963911e-02]
#  [8.20737302e-01 1.48528107e-02 1.37239359e-02 3.36842909e-02
#   4.23880741e-02 2.12493539e-02 3.37334909e-02 1.96306892e-02]
#  [9.80194569e-01 2.54904182e-04 1.73930210e-04 3.51769442e-04
#   2.41371919e-03 1.14897825e-02 1.77387486e-03 3.34733725e-03]
#  [9.88472641e-01 3.29190734e-05 1.57506565e-05 7.35595313e-05                           
#  1.03055045e-03 6.18949812e-03 1.13323855e-03 3.05172033e-03]
#  [9.97202635e-01 3.42596250e-05 1.43712987e-05 1.46676657e-05 
#  4.98445996e-04 7.32441491e-04 1.59672098e-04 1.34355435e-03]
#  [9.99532700e-01 2.92500099e-05 1.01696041e-05 3.75845389e-06
#   1.03849030e-04 1.71033214e-04 5.76570019e-05 9.15866840e-05]]

# # maxpooling 倾向的分布
# [[0.36193225 0.4379051  0.0161191  0.02069746 0.02797149 0.03840489
#       0.02190726 0.07506248]
#      [0.14874698 0.4382021  0.00555205 0.00456976 0.01933626 0.2138325
#       0.05921678 0.11054359]
#      [0.25426775 0.5625067  0.03110873 0.03069633 0.03765448 0.03093559
#       0.01902944 0.0338011 ]
#      [0.18767262 0.60414505 0.00812492 0.00974984 0.01632629 0.03341402
#       0.05659105 0.08397621]
#      [0.14125268 0.7334557  0.01533368 0.00646582 0.01605983 0.02090663
#       0.00438534 0.06214031]
#      [0.17075664 0.32027614 0.04807629 0.04313296 0.03628872 0.08493719
#       0.13115026 0.16538177]
#      [0.28165826 0.38778698 0.02194643 0.03327375 0.03618929 0.0749488
#       0.08338676 0.08080978]
#      [0.18221524 0.5563337  0.03731598 0.02879707 0.07023829 0.02768439
#       0.02906159 0.06835366]
#      [0.12437709 0.7878268  0.02589052 0.00713319 0.0063407  0.01732855
#       0.01846603 0.01263708]
#      [0.27924806 0.12657717 0.05513798 0.10103597 0.06939318 0.09558422
#       0.10619463 0.16682878]
#      [0.39360607 0.16481358 0.03322955 0.05694548 0.08762505 0.06722024
#       0.07258383 0.12397628]
#      [0.34780723 0.20023367 0.03774906 0.04975709 0.11062324 0.06319663
#       0.08460145 0.10603157]
#      [0.3756192  0.46067354 0.03725033 0.0136018  0.02810322 0.02580596
#       0.03471563 0.02423032]
#      [0.18215683 0.7414551  0.05356305 0.00316706 0.00615716 0.00508859
#       0.00446206 0.00395018]


# skip connect 倾向的分布











BATS = Genotype(normal=[('group_conv_5x5', 0), ('dil_group_conv_5x5', 1), 
                        ('max_pool_3x3', 0),('max_pool_3x3', 2),
                        ('group_conv_5x5', 0), ('dil_group_conv_5x5', 2), 
                        ('group_conv_3x3',0), ('dil_group_conv_5x5', 1)], 
                        normal_concat=range(2, 6), 
                reduce=[('max_pool_3x3', 0), ('group_conv_3x3', 1), 
                        ('group_conv_5x5', 0), ('avg_pool_3x3', 1), 
                        ('group_conv_5x5', 1), ('max_pool_3x3', 3), 
                        ('group_conv_5x5', 0), ('group_conv_5x5', 1)], 
                        reduce_concat=range(2, 6))

# 5个5 2个3
L1cost = Genotype(normal=[('dil_group_conv_5x5', 0), ('dil_group_conv_3x3', 1), 
                         ('dil_group_conv_5x5', 2), ('group_conv_5x5', 0), 
                         ('dil_group_conv_5x5', 2), ('avg_pool_3x3', 0), 
                         ('group_conv_3x3', 0), ('group_conv_5x5', 1)], 
                        normal_concat=range(2, 6), 
                reduce=[('avg_pool_3x3', 1), ('max_pool_3x3', 0), 
                        ('avg_pool_3x3', 1), ('max_pool_3x3', 0), 
                        ('dil_group_conv_5x5', 3), ('dil_group_conv_5x5', 2), 
                        ('dil_group_conv_5x5', 3), ('dil_group_conv_5x5', 2)], 
                        reduce_concat=range(2, 6))

# 3个5 3个3
L2cost = Genotype(normal=[('group_conv_3x3', 0), ('dil_group_conv_3x3', 1), 
                            ('skip_connect', 0), ('dil_group_conv_5x5', 2), 
                            ('group_conv_3x3', 0), ('max_pool_3x3', 2), 
                            ('dil_group_conv_3x3', 0), ('dil_group_conv_5x5', 2)], 
                        normal_concat=range(2, 6), 

                reduce=[('dil_group_conv_3x3', 0), ('max_pool_3x3', 1), 
                        ('avg_pool_3x3', 0), ('dil_group_conv_3x3', 1), 
                        ('max_pool_3x3', 0), ('dil_group_conv_5x5', 1), 
                        ('max_pool_3x3', 4), ('skip_connect', 0)], 
                        reduce_concat=range(2, 6))


# 4个5 2个3 2个pool
L3cost = Genotype(normal=[('dil_group_conv_3x3', 1), ('dil_group_conv_5x5', 0), 
                        ('dil_group_conv_5x5', 1), ('dil_group_conv_3x3', 0), 
                        ('dil_group_conv_5x5', 3), ('avg_pool_3x3', 1), 
                        ('max_pool_3x3', 1), ('dil_group_conv_5x5', 4)], 
                        normal_concat=range(2, 6), 
                reduce=[('max_pool_3x3', 1), ('group_conv_3x3', 0), 
                        ('avg_pool_3x3', 1), ('dil_group_conv_5x5', 0), 
                        ('avg_pool_3x3', 1), ('dil_group_conv_5x5', 2), 
                        ('dil_group_conv_3x3', 1), ('dil_group_conv_3x3', 0)], 
                        reduce_concat=range(2, 6))



L1reduce2normal =  Genotype(normal=[('group_conv_3x3', 0), ('dil_group_conv_3x3', 1), 
                            ('skip_connect', 0), ('dil_group_conv_5x5', 2), 
                            ('group_conv_3x3', 0), ('max_pool_3x3', 2), 
                            ('dil_group_conv_3x3', 0), ('dil_group_conv_5x5', 2)], 
                        normal_concat=range(2, 6), 

                        reduce=[('avg_pool_3x3', 1), ('max_pool_3x3', 0), 
                                ('avg_pool_3x3', 1), ('max_pool_3x3', 0), 
                                ('dil_group_conv_5x5', 3), ('dil_group_conv_5x5', 2), 
                                ('dil_group_conv_5x5', 3), ('dil_group_conv_5x5', 2)], 
                                reduce_concat=range(2, 6))


# 3个5 3个3
L2cost_v2 = Genotype(normal=[('group_conv_3x3', 0), ('dil_group_conv_3x3', 1), 
                            ('skip_connect', 0), ('dil_group_conv_5x5', 2), 
                            ('dil_group_conv_5x5', 0), ('max_pool_3x3', 2), 
                            ('dil_group_conv_3x3', 0), ('dil_group_conv_5x5', 2)], 
                        normal_concat=range(2, 6), 

                reduce=[('dil_group_conv_3x3', 0), ('max_pool_3x3', 1), 
                        ('avg_pool_3x3', 0), ('dil_group_conv_3x3', 1), 
                        ('max_pool_3x3', 0), ('dil_group_conv_5x5', 1), 
                        ('max_pool_3x3', 4), ('skip_connect', 0)], 
                        reduce_concat=range(2, 6))

# 3个5 3个3 全 dil 
L2cost_v3 = Genotype(normal=[('dil_group_conv_3x3', 0), ('dil_group_conv_3x3', 1), 
                            ('skip_connect', 0), ('dil_group_conv_5x5', 2), 
                            ('dil_group_conv_5x5', 0), ('max_pool_3x3', 2), 
                            ('dil_group_conv_3x3', 0), ('dil_group_conv_5x5', 2)], 
                        normal_concat=range(2, 6), 

                reduce=[('dil_group_conv_3x3', 0), ('max_pool_3x3', 1), 
                        ('avg_pool_3x3', 0), ('dil_group_conv_3x3', 1), 
                        ('max_pool_3x3', 0), ('dil_group_conv_5x5', 1), 
                        ('max_pool_3x3', 4), ('skip_connect', 0)], 
                        reduce_concat=range(2, 6))



adv5_drop123t02_15 = Genotype(normal=[('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 1), ('dil_group_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_group_conv_3x3', 1), ('dil_group_conv_5x5', 3), ('max_pool_3x3', 1), ('group_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('group_conv_3x3', 0), ('avg_pool_3x3', 1), ('group_conv_3x3', 0), ('group_conv_5x5', 1), ('group_conv_5x5', 2), ('skip_connect', 1), ('skip_connect', 0)], reduce_concat=range(2, 6))

adv5_drop123t02_19 = Genotype(normal=[('dil_group_conv_5x5', 1), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 1), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 3), ('dil_group_conv_3x3', 1), ('max_pool_3x3', 1), ('dil_group_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('group_conv_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('dil_group_conv_5x5', 2), ('skip_connect', 1), ('skip_connect', 0)], reduce_concat=range(2, 6))

adv5_drop123t02_24 = Genotype(normal=[('dil_group_conv_5x5', 0), ('max_pool_3x3', 1), ('dil_group_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_group_conv_5x5', 3), ('max_pool_3x3', 0), ('dil_group_conv_5x5', 4), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('group_conv_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('group_conv_5x5', 2), ('skip_connect', 1), ('group_conv_3x3', 2)], reduce_concat=range(2, 6))




#  new   drop0  t0.2  react-wop  adv2 
stage0 =  Genotype(normal=[('group_conv_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('dil_group_conv_3x3', 1), ('max_pool_3x3', 3), ('skip_connect', 2), ('dil_group_conv_3x3', 2), ('dil_group_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('group_conv_3x3', 1), ('group_conv_5x5', 2), ('dil_group_conv_3x3', 0), ('skip_connect', 3), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 2), ('skip_connect', 1)], reduce_concat=range(2, 6))
stage1 =  Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3',2), ('dil_group_conv_3x3', 1), ('max_pool_3x3', 3), ('group_conv_5x5', 2), ('dil_group_conv_5x5', 4), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('dil_group_conv_5x5', 1), ('skip_connect', 0),('dil_group_conv_5x5', 2), ('dil_group_conv_5x5', 0), ('skip_connect', 0), ('dil_group_conv_5x5', 2), ('avg_pool_3x3', 4), ('avg_pool_3x3', 3)], reduce_concat=range(2, 6))
adv_stage3 =  Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 3), ('dil_group_conv_3x3', 0), ('dil_group_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('dil_group_conv_5x5', 1), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 2), ('group_conv_5x5',1), ('dil_group_conv_3x3', 3), ('group_conv_5x5', 0), ('group_conv_5x5', 2)], reduce_concat=range(2, 6))

#  new   drop0  t0.2  react-wop  no adv 
stage3_15 = Genotype(normal=[('group_conv_5x5', 0), ('dil_group_conv_5x5', 1), ('dil_group_conv_5x5', 1), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 2), ('max_pool_3x3', 0), ('group_conv_5x5', 2), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('dil_group_conv_5x5', 0), ('avg_pool_3x3', 1), ('dil_group_conv_5x5', 2), ('max_pool_3x3', 3), ('group_conv_5x5', 2), ('avg_pool_3x3', 1), ('dil_group_conv_5x5', 2)], reduce_concat=range(2, 6))
stage3_25 = Genotype(normal=[('dil_group_conv_5x5', 1), ('group_conv_5x5', 0), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 1), ('dil_group_conv_5x5', 2), ('dil_group_conv_5x5', 3), ('skip_connect', 0), ('max_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('dil_group_conv_5x5', 0), ('group_conv_3x3', 0), ('dil_group_conv_5x5', 2), ('max_pool_3x3', 3), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))
noadv_stage3 =  Genotype(normal=[('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 1), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 1), ('dil_group_conv_5x5', 2), ('dil_group_conv_5x5', 3), ('skip_connect', 0), ('max_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[('dil_group_conv_5x5', 0), ('avg_pool_3x3', 1), ('group_conv_3x3', 0), ('dil_group_conv_5x5', 2), ('dil_group_conv_5x5', 2), ('max_pool_3x3', 3), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))

#  new   drop0  t0.2  react-wop  adv2 
waadv_stage2_13 = Genotype(normal=[('group_conv_5x5', 0), ('dil_group_conv_5x5', 1), ('group_conv_5x5', 2), ('dil_group_conv_5x5', 1), ('dil_group_conv_5x5', 3), ('max_pool_3x3', 1), ('group_conv_5x5', 1), ('dil_group_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('group_conv_5x5', 0), ('dil_group_conv_5x5', 2), ('group_conv_5x5', 1), ('dil_group_conv_5x5', 2), ('dil_group_conv_5x5', 3), ('max_pool_3x3', 4), ('dil_group_conv_5x5', 2)], reduce_concat=range(2, 6))

waadv_stage3_11 = Genotype(normal=[('dil_group_conv_5x5', 1), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 2), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 2), ('skip_connect', 0), ('dil_group_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 2), ('avg_pool_3x3', 0), ('dil_group_conv_5x5', 2), ('group_conv_5x5', 0), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

waadv_stage3_12 = Genotype(normal=[('dil_group_conv_5x5', 1), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 2), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 1), ('skip_connect', 0), ('dil_group_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 2), ('avg_pool_3x3', 0), ('dil_group_conv_5x5', 2), ('group_conv_5x5', 0), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))


waadv_stage3_13 = Genotype(normal=[('dil_group_conv_5x5', 1), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 2), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 1), ('skip_connect', 0), ('dil_group_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 2), ('avg_pool_3x3', 0), ('dil_group_conv_5x5', 2), ('group_conv_5x5', 0), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

waadv_stage3_15 = Genotype(normal=[('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 1), ('dil_group_conv_5x5',0), ('dil_group_conv_5x5', 2), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 1), ('skip_connect', 0), ('dil_group_conv_5x5', 4)], normal_concat=range(2, 6),
                    reduce=[('dil_group_conv_5x5', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_group_conv_5x5', 2), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 4)], reduce_concat=range(2, 6))

waadv_stage3_15_v2 = Genotype(normal=[('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 1), ('skip_connect',0), ('dil_group_conv_5x5', 2), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 1), ('skip_connect', 0), ('dil_group_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('dil_group_conv_5x5', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_group_conv_5x5', 2), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 4)], reduce_concat=range(2, 6))

waadv_stage3_15_v3 = Genotype(normal=[('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 1), 
                                        ('skip_connect',0), ('dil_group_conv_5x5', 2), 
                                        ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 1), 
                                        ('skip_connect', 0), ('dil_group_conv_5x5', 4)], 
                                        normal_concat=range(2, 6), 
                                reduce=[('dil_group_conv_3x3', 0), ('max_pool_3x3', 1), 
                                        ('avg_pool_3x3', 0), ('dil_group_conv_3x3', 1), 
                                        ('max_pool_3x3', 0), ('dil_group_conv_5x5', 1), 
                                        ('max_pool_3x3', 4), ('skip_connect', 0)], 
                                        reduce_concat=range(2, 6))

waadv_stage3_final = Genotype(normal=[('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 1), 
                                        ('max_pool_3x3',0), ('dil_group_conv_5x5', 2), 
                                        ('dil_group_conv_5x5', 0), ('dil_group_conv_3x3', 1), 
                                        ('skip_connect', 0), ('dil_group_conv_5x5', 4)], 
                                        normal_concat=range(2, 6), 
                                            reduce=[('dil_group_conv_5x5', 0), ('avg_pool_3x3', 1), 
                                            ('avg_pool_3x3', 0), ('dil_group_conv_5x5', 2), 
                                            ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 2), 
                                            ('max_pool_3x3', 3), ('max_pool_3x3', 4)], 
                                        reduce_concat=range(2, 6))

