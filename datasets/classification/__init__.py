# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-09-15 23:58:45
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-10-03 22:01:22

from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from .transform  import augmentation as tf
from .paths import path
from .cifar10 import CIFAR10
from .cifar100 import CIFAR100
from .mnist import MNIST
from .stl10 import STL10
from .svhn import SVHN
from .imagenet import ImageFolder
from .tiny_imagenet import TinyImagenet

datasets_dict={
    "cifar10":CIFAR10,
    "cifar100":CIFAR100,
    "mnist":MNIST,
    "stl10":STL10,
    "svhn":SVHN,
    "imagenet":ImageFolder,
    "tiny_imagenet": TinyImagenet
}


def getTransform(dataset,flag,cutout=False):
    assert dataset in datasets_dict.keys(),"the dataset is not support!"
    if dataset in ["cifar10","cifar100","svhn"]:
        if flag=="train":
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32,padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),std=(0.2023, 0.1994, 0.2010))
                ])
        else:
            transform=tf.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),std=(0.2023, 0.1994, 0.2010))
                ])

    elif dataset=="imagenet":
        if flag=="train":
            transform =transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                         transforms.ColorJitter(brightness=0.5,
                                                                contrast=0.5,
                                                                saturation=0.3),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
        else:
            transform =transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
    elif dataset=="tiny_imagenet":
        if flag=="train":
            transform =transforms.Compose([
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomCrop(64,padding=8),
                                         transforms.ColorJitter(brightness=0.5,
                                                                contrast=0.5,
                                                                saturation=0.3),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                         ])
        else:
            transform =transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                         ])
    elif dataset=="mnist":

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.1307], std=[0.3081])
             ])

    elif dataset=="stl10":
        transform=None

    else:
        print('DatasetTransform {} not available.'.format(dataset))
        raise NotImplementedError

    if cutout:
        if dataset == "imagenet":
            length = 112
        elif dataset == "tiny_imagenet":
            length = 32
        elif dataset in  ["cifar10","cifar100","svhn"]:
            length = 16
        else:
            length = 8
        transform.transforms.append(tf.Cutout(length=length,n_holes=1))

    return transform


def getDataloader(dataset,flag,batch_size,num_workers,pin_memory,cutout=False):
    '''
    return args.dataset`s dataloader
    '''
    root=path.db_root_dir(dataset)
    trans=getTransform(dataset,flag,cutout=cutout)
    assert root is not None and trans is not None
    tflag = True if flag=="train" else False
    if dataset=="imagenet":
        new_root=os.path.join(root,flag)
        Data=datasets_dict[dataset](root=new_root,transform=trans)
    else:
        Data=datasets_dict[dataset](root=root,train=tflag,download=True,transform=trans)

    if flag=="train":
        shuffle=True
    else:
        shuffle=False

    return DataLoader(dataset=Data,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=True)


if __name__ == '__main__':
    import argparse
    from PIL import  Image
    import numpy as np
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.dataset='tiny_imagenet'
    args.train_batch=1
    args.val_batch=1
    args.num_workers=1
    args.flag="train"
    args.pin_memory=True
    args.cutout=True
    dataloader=getDataloader(args.dataset,args.flag,args.train_batch,args.num_workers,args.pin_memory,
                            args.cutout)

    count=0
    for i,sample in enumerate(dataloader):
        # n c h w     n 
        images,labels=sample
        print(images.size(),labels.size())
        image=images.numpy()[0].transpose(1,2,0)
        image=image*255
        image=Image.fromarray(image.astype(np.uint8))
        image.show("Image")
        print(labels.numpy()[0])
        count+=1
        if count>10:
            break

