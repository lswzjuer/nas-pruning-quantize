# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-09-16 13:24:49
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-09-16 16:48:36
import numpy as np 
from PIL import Image
import torch 
import torchvision.transforms as transforms
import torchvision.transforms.functional as trans_func


class Compose(object):
    """docstring for Compose"""
    def __init__(self, transformList):
        super(Compose, self).__init__()
        self.transforms = transformList

    def __call__(self,image):
        for t in self.transforms:
            image=t(image)
        return image


class RandomCrop(object):
    """docstring for RandomCrop"""
    def __init__(self, cropsize,padding):
        super(RandomCrop, self).__init__()
        self.cropsize = cropsize
        self.padding=padding

    def __call__(self,image):
        '''
        input is Image type
        '''
        return transforms.RandomCrop(self.cropsize,self.padding)(image)


class RandomHorizontalFlip(object):
    """docstring for ClassName"""
    def __init__(self):
        super(RandomHorizontalFlip, self).__init__()

    def __call__(self,image):
        return transforms.RandomHorizontalFlip()(image)


class ToTensorNormalize(object):
    """docstring for ToTensorNormalize"""
    def __init__(self, mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]):
        super(ToTensorNormalize, self).__init__()
        self.mean = mean
        self.std=std

    def __call__(self,image):
        image=transforms.ToTensor()(image)
        image=transforms.Normalize(self.mean,self.std)(image)
        return image


class Cutout(object):
    def __init__(self, length,n_holes=1):
        super(Cutout, self).__init__()
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img








