# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-09-15 23:58:37
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-09-16 16:59:05

from torch.utils.data import Dataset
import scipy.io as sio
import os 
import numpy as np 


class SVHN(Dataset):
    """docstring for svhn"""
    def __init__(self,root,train=True,download=False,transform=None,target_transform=None):
        super(svhn, self).__init__()
        self.root=root
        self.transform=transform
        self.target_transform=target_transform
        self.train=train
        if self.train:
            data_file="train_32x32.mat"
        else:
            data_file="test_32x32.mat"
        loaded_mat=sio.loadmat(os.path.join(self.root,data_file))          
        self.data = loaded_mat['X']
        self.targets = loaded_mat['y'].astype(np.int64).squeeze()
        # SVHN assigns the class label "10" to the digit 0
        # change the class labels to be in the range [0, C-1]
        np.place(self.targets, self.targets == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)




# def SVHN(data_root, batch_size, num_workers, **kwargs):

#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#     ])
#     testset = SVHN_Dataset(False, transform=transform_test)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#     return testloader
