# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-09-16 14:06:28
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-09-18 00:35:01

class path(object):
    """docstring for paths"""
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'cifar10':
            return r'F:\source_code\DataSets'
        elif dataset == 'cifar100':
            return r'F:\source_code\DataSets'
        elif dataset == 'svhn':
            return r'F:\source_code\DataSets'
        elif dataset == 'imagenet':
            return r'F:\source_code\DataSets'
        elif dataset == 'tiny_imagenet':
            return r'G:\codeing\DataSets\tiny-imagenet-200'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
        