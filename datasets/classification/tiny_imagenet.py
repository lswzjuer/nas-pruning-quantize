# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-09-17 23:00:52
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-09-18 00:37:33


import numpy as np 
import torch
import os 
from torch.utils.data import Dataset
import cv2
from PIL import Image




def gerTxtFile(path):
    folder_labels={}
    count=0
    with open(path,"r") as f:
        for line in f.readlines():
            folder_name=line.strip("\n")
            folder_labels[folder_name]=count
            count+=1
    assert count==200 and len(folder_labels)==200
    return folder_labels


def gerTxtFileVal(path):
    names=[]
    folders=[]
    with open(path,"r") as f:
        for line in f.readlines():
            val_name = line.strip('\n').split('\t')[0]
            val_folder = line.strip('\n').split('\t')[1]
            names.append(val_name)
            folders.append(val_folder)
    return names,folders




class TinyImagenet(object):
    """docstring for TinyImagenet"""
    def __init__(self, root,train=True,download=False,transform=None):
        super(TinyImagenet, self).__init__()
        assert os.path.exists(root)
        self.root=root
        self.train=train
        self.transform=transform

        self.train_fo=os.path.join(self.root,"train")
        self.val_fo=os.path.join(self.root,"val")

        self.labelfile=os.path.join(self.root,"wnids.txt")
        self.labeldict=gerTxtFile(self.labelfile)

        if self.train:
            self.images,self.labels=self.getImagesTrain()
        else:
            self.images,self.labels=self.getImagesVal()

    def getImagesTrain(self):
        images=[]
        labels=[]
        imagefolders=os.listdir(self.train_fo)
        for fo in imagefolders:
            label=self.labeldict[fo]
            path=os.path.join(self.train_fo,fo,"images")
            subimages=[os.path.join(path,file) for file in os.listdir(path)]
            sublabels=[label]*len(subimages)
            # print("name:{} label:{},imagenum:{}".format(fo,label,len(subimages)))
            images+=subimages
            labels+=sublabels
        assert len(images)==len(labels)
        return images,labels


    def getImagesVal(self):
        labelfile=os.path.join(self.val_fo,"val_annotations.txt")
        names,folders=gerTxtFileVal(labelfile)
        images=[os.path.join(self.val_fo,"images",file) for file in names]
        labels=[self.labeldict[fo] for fo in folders]
        return images,labels

    def __getitem__(self,index):
        image,label=self.images[index],self.labels[index]
        # read image
        image = Image.open(image).convert("RGB")
        # image = image[:, :, ::-1]
        if self.transform:
            image=self.transform(image)
        return image,label

    def __len__(self):
        return len(self.images)




if __name__ == '__main__':
    root=r"G:\codeing\DataSets\tiny-imagenet-200"
    imagenet=TinyImagenet(root)


