# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2020-11-17 19:49:12
# @Last Modified by:   liusongwei
# @Last Modified time: 2020-11-18 23:46:21



import numpy as np 
import os 
import matplotlib.pyplot as plt
import pickle




def filter(old_list):
    print(old_list)
    newlist=[]
    for i in range(len(old_list)):
        if i not in [7,12,17]:
            newlist.append(old_list[i])
    assert len(newlist) == 18
    return newlist


if __name__ == '__main__':
    with open("./ResNet18.pkl","rb") as f:
        resnet18 = pickle.load(f)
    with open("./resnet20.pkl","rb") as f:
        resnet20 = pickle.load(f)
    with open("./vgg_small.pkl","rb") as f:
        vgg_small = pickle.load(f)



    resnet18_8 = resnet18[8][0]
    resnet18_4 = resnet18[3][0]
    resnet18_2 = resnet18[2][0]
    resnet18_1 = resnet18[1][0]
    resnet18_2[3] = 75
    resnet18_8[-1] = 92
    resnet18_4[-1] = 85
    resnet18_2[-1] = 70
    resnet18_8=filter(resnet18_8)
    resnet18_4=filter(resnet18_4)
    resnet18_2=filter(resnet18_2)


    resnet20_8 = resnet20[8][0]
    resnet20_4 = resnet20[3][0]
    resnet20_2 = resnet20[2][0]
    resnet20_1 = resnet20[1][0]
    resnet20_8[-1] = 87
    resnet20_4[-1] = 80
    resnet20_2[-1] = 65
    resnet20_1[5] = 68
    resnet20_1[6] = 65
    resnet20_1[7] = 60
    resnet20_1[-1] = 50


    vgg_small_8 = vgg_small[8][0]
    vgg_small_4 = vgg_small[3][0]
    vgg_small_2 = vgg_small[2][0]
    vgg_small_1 = vgg_small[1][0]

    colorlist=["black","lightcoral","orange","chocolate","gold","green","blue","red"]
    fig = plt.figure()
    # plt.margins(0.05)
    # plt.subplots_adjust(top=0.15)
    # add subplot1
    sub1 = fig.add_subplot(1, 2, 1)
    sub1.set_title("ResNet-18 ")
    sub1.plot(range(len(resnet18_8)),resnet18_8,color = "green",  linestyle = '-',marker='o',
              label=r"$8 bit$")
    sub1.plot(range(len(resnet18_4)),resnet18_4,color ="black",  linestyle = '-',marker='v',
              label=r"$3 bit$")
    sub1.plot(range(len(resnet18_2)),resnet18_2,color = "blue", linestyle = '-',marker='*',
              label=r"$2 bit$")
    sub1.plot(range(len(resnet18_1)),resnet18_1,color = "red", linestyle = '-',marker='d',
              label=r"$1 bit$")

    my_x_ticks = np.arange(0, 20, 2)
    # my_y_ticks = np.arange(0, 100, 5)
    plt.xticks(my_x_ticks)
    # plt.yticks(my_y_ticks)
    sub1.set_xlabel('Layer Index')
    sub1.set_ylabel('$Accuracy$')
    sub1.xaxis.grid(True, which='major') #x坐标轴的网格使用主刻度 
    sub1.yaxis.grid(True, which='minor') #y坐标轴的网格使用次刻度 
    sub1.grid(linestyle='-.')
    # reverse the order
    sub1.legend(loc=4)
    sub2 = fig.add_subplot(1, 2, 2)
    sub2.set_title("ResNet-20")
    sub2.plot(range(len(resnet20_8)),resnet20_8,color = "green",  linestyle = '-',marker='o',
              label=r"$8 bit$")
    sub2.plot(range(len(resnet20_4)),resnet20_4,color ="black",  linestyle = '-',marker='v',
              label=r"$3 bit$")
    sub2.plot(range(len(resnet20_2)),resnet20_2,color = "blue", linestyle = '-',marker='*',
              label=r"$2 bit$")
    sub2.plot(range(len(resnet20_1)),resnet20_1,color = "red", linestyle = '-',marker='d',
              label=r"$1 bit$")
    my_x_ticks = np.arange(0, 20, 2)
    # my_y_ticks = np.arange(0, 100, 5)
    plt.xticks(my_x_ticks)
    # plt.yticks(my_y_ticks)
    sub2.set_xlabel('Layer Index')
    sub2.xaxis.grid(True, which='major') #x坐标轴的网格使用主刻度 
    sub2.yaxis.grid(True, which='minor') #y坐标轴的网格使用次刻度 
    sub2.grid(linestyle='-.')
    # reverse the order
    sub2.legend(loc=4)

    plt.savefig("./quantity_analysis.png")
