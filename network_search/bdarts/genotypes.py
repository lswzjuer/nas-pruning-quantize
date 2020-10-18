# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-10-13 23:39:15
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-10-15 15:37:07

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