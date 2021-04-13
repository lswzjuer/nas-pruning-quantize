# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-12-14 23:20:46
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-12-14 23:38:44

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib

plt.rc('font',family='Times New Roman')
del matplotlib.font_manager.weight_dict['roman']
matplotlib.font_manager._rebuild()

# ndarray
def sign(dlist):
    dlist[dlist<0] = -1
    dlist[dlist>=0] = 1
    return dlist

def htanh(dlist):
    dlist[dlist<=-1] = -1
    dlist[dlist>=1] = 1
    return dlist


def approx(dlist):
    mask1 = dlist < -1
    mask2 = dlist < 0
    mask3 = dlist < 1
    mask = -1 * mask1.astype(np.float32) + (dlist * 2 + np.power(dlist,2)) * (1 - mask1.astype(np.float32))
    mask = mask * mask2.astype(np.float32) + (dlist * 2 - np.power(dlist,2)) * (1 - mask2.astype(np.float32))
    mask = mask * mask3.astype(np.float32) + 1 * (1 - mask3.astype(np.float32))
    return mask



def tanhpx(dlist,alpha=1):

    px=np.exp(alpha*dlist)
    fpx=np.exp(-alpha*dlist)
    res = (px-fpx)/(px+fpx)
    return res

def sigmod(dlist):
    px=np.exp(dlist*(-1))
    return 1/(1 + px)


def relu(dlist):
    mask = dlist < 0
    return 0*mask.astype(np.float32) + dlist*(1- mask.astype(np.float32))




if __name__ == '__main__':

    flist = np.linspace(start=-2,stop=2,num=2000)
    flist = np.asarray(flist)

    bluelist=['lavender',"lightsteelblue",'cornflowerblue','royalblue','blue','mediumblue','darkblue','navy']
    fig = plt.figure()

    # add subplot1
    sub1 = fig.add_subplot(1, 3, 1)
    sub1.set_title("Sigmod", fontdict={'weight': 'normal', 'size': 13})
    sub1.plot(flist,sigmod(flist),color = 'red', linewidth = 2.0, linestyle = '-')
    sub1.set_xlim(-2, 2)
    sub1.set_ylim(-3, 3)
    sub1.set_xlabel('x', fontdict={'weight': 'normal', 'size': 15})
    sub1.set_ylabel('f(x)',fontdict={'weight': 'normal', 'size': 15})
    sub1.grid(linestyle='-.')


    sub2 = fig.add_subplot(1, 3,2)
    sub2.set_title("Tanh", fontdict={'weight': 'normal', 'size': 13})
    sub2.plot(flist,tanhpx(flist),color = 'red', linewidth = 2.0, linestyle = '-')
    sub2.set_xlim(-2, 2)
    sub2.set_ylim(-3, 3)
    sub2.set_xlabel('x', fontdict={'weight': 'normal', 'size': 15})
    # sub2.set_ylabel('$f(x)$')
    sub2.grid(linestyle='-.')


    sub3 = fig.add_subplot(1, 3, 3)
    sub3.set_title("ReLu", fontdict={'weight': 'normal', 'size': 13})
    sub3.plot(flist,relu(flist),color = 'red', linewidth = 2.0, linestyle = '-')
    sub3.set_xlim(-2, 2)
    sub3.set_ylim(-3, 3)
    sub3.set_xlabel('x', fontdict={'weight': 'normal', 'size': 15})
    # sub2.set_ylabel('$f(x)$')
    sub3.grid(linestyle='-.')

    plt.show()