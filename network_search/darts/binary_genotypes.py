# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-10-13 23:39:15
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-11-29 21:09:15

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


BATS_DROP245_LR025_F10_ic18gc6convshuffleres_nott_stage3 = Genotype(normal=[('group_conv_3x3', 0), ('dil_group_conv_5x5', 1), ('group_conv_3x3', 0), ('group_conv_5x5', 1), ('group_conv_3x3', 0), ('group_conv_5x5', 1), ('dil_group_conv_5x5', 0), ('group_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('max_pool_3x3', 4)], reduce_concat=range(2, 6))

BATS_DROP245_LR025_F10_ic18gc6convshuffleres_nott_stage2 = Genotype(normal=[('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 1), ('group_conv_3x3', 1), ('dil_group_conv_5x5', 2), ('dil_group_conv_5x5', 0), ('max_pool_3x3', 1), ('group_conv_5x5', 1), ('dil_group_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('group_conv_5x5', 0), ('skip_connect', 1), ('max_pool_3x3', 3), ('dil_group_conv_5x5', 0)], reduce_concat=range(2, 6))

BATS_DROP245_LR001_F10_ic18gc6convshuffleres_nott_stage3 = Genotype(normal=[('max_pool_3x3', 0), ('dil_group_conv_5x5', 1), ('dil_group_conv_5x5', 0), ('dil_group_conv_5x5', 1), ('dil_group_conv_5x5', 1), ('dil_group_conv_5x5', 2), ('dil_group_conv_5x5', 0), ('dil_group_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('group_conv_3x3', 0), ('dil_group_conv_5x5', 2), ('group_conv_5x5', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))

BATS_DROP245_LR001_F10_ic18gc6convshuffleres_nott_stage2 =  Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('group_conv_5x5', 0), ('max_pool_3x3', 3), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

